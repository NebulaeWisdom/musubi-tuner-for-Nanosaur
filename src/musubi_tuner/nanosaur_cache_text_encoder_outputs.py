import argparse
import logging
import os

import torch
from safetensors.torch import save_file

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_NANOSAUR,
    ItemInfo,
    save_text_encoder_output_cache_nanosaur,
)
from musubi_tuner.nanosaur_utils import build_text_encoder, encode_text
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device: torch.device):
    prompts = [item.caption for item in batch]
    prompt_embeds = encode_text(prompts, tokenizer, text_encoder, device)
    for item, embed in zip(batch, prompt_embeds):
        save_text_encoder_output_cache_nanosaur(item, embed)


def save_uncond_embedding(tokenizer, text_encoder, device: torch.device, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "uncond_ns_te.safetensors")
    embed = encode_text([""], tokenizer, text_encoder, device)[0]
    save_file({"uncond_gemma_embed_float16": embed.detach().cpu().contiguous()}, path, metadata={"architecture": "nanosaur"})
    logger.info(f"Saved unconditional NanoSaur text embedding: {path}")


def nanosaur_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="NanoSaur text encoder checkpoint path")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for text encoder, default is float16")
    parser.add_argument("--skip_uncond", action="store_true", help="do not write uncond_ns_te.safetensors")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = nanosaur_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_NANOSAUR)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    logger.info(f"Loading NanoSaur text encoder from {args.text_encoder}")
    tokenizer, text_encoder = build_text_encoder(device=device, dtype=text_encoder_dtype, checkpoint_path=args.text_encoder)

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(tokenizer, text_encoder, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)

    if not args.skip_uncond:
        cache_dirs = {dataset.cache_directory for dataset in datasets if dataset.cache_directory is not None}
        for cache_dir in cache_dirs:
            save_uncond_embedding(tokenizer, text_encoder, device, cache_dir)

    del tokenizer, text_encoder
    logger.info("Done!")


if __name__ == "__main__":
    main()
