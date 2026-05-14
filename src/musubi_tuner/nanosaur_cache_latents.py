import logging
from typing import List

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_NANOSAUR, ItemInfo, save_latent_cache_nanosaur
from musubi_tuner.nanosaur_utils import NanoSaurVAEWrapper
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_nanosaur(batch: List[ItemInfo]) -> torch.Tensor:
    contents = []
    for item in batch:
        content = torch.from_numpy(item.content)
        if content.shape[-1] == 4:
            content = content[..., :3]
        contents.append(content)
    contents = torch.stack(contents, dim=0)
    contents = contents.permute(0, 3, 1, 2).contiguous()
    return contents.float() / 127.5 - 1.0


def encode_and_save_batch(vae: NanoSaurVAEWrapper, batch: List[ItemInfo]):
    contents = preprocess_contents_nanosaur(batch)
    h, w = contents.shape[2], contents.shape[3]
    if h < 16 or w < 16:
        item = batch[0]
        raise ValueError(f"Image size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    latents = vae.encode(contents)
    for item, latent in zip(batch, latents):
        save_latent_cache_nanosaur(item_info=item, latent=latent)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_NANOSAUR)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1)
        return

    if args.vae is None:
        raise ValueError("VAE checkpoint is required (--vae)")

    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    logger.info(f"Loading NanoSaur VAE from {args.vae}")
    vae = NanoSaurVAEWrapper(args.vae, device=device, dtype=vae_dtype)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)
    logger.info("Done!")


if __name__ == "__main__":
    main()
