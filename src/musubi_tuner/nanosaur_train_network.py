import argparse
import logging

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_NANOSAUR, ARCHITECTURE_NANOSAUR_FULL
from musubi_tuner.hv_train_network import NetworkTrainer, hv_setup_parser, read_config_from_file, setup_parser_common
from musubi_tuner.nanosaur_utils import NanoSaurVAEWrapper, load_nanosaur_transformer, sample_timesteps
from musubi_tuner.utils import model_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NanoSaurNetworkTrainer(NetworkTrainer):
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_NANOSAUR

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_NANOSAUR_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        if args.dit_dtype is None:
            args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 0.0
        self.default_discrete_flow_shift = 4.0
        self.uncond_text_embedding = None
        self.timestep_sampling_alpha = args.timestep_sampling_alpha
        self.cond_dropout = args.cond_dropout
        if not args.dit:
            raise ValueError("--dit is required for NanoSaur training")
        if args.blocks_to_swap:
            raise ValueError("NanoSaur minimal musubi support does not implement --blocks_to_swap yet")
        if args.fp8_base:
            raise ValueError("NanoSaur minimal musubi support does not implement --fp8_base yet")

    @property
    def i2v_training(self) -> bool:
        return False

    @property
    def control_training(self) -> bool:
        return False

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module):
        return weights_sd

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        raise NotImplementedError("NanoSaur minimal musubi support does not implement sampling yet. Omit --sample_prompts.")

    def do_inference(self, *args, **kwargs):
        raise NotImplementedError("NanoSaur minimal musubi support does not implement sampling yet.")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        return NanoSaurVAEWrapper(vae_path, device="cpu", dtype=vae_dtype)

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        dtype = dit_weight_dtype if dit_weight_dtype is not None else self.dit_dtype
        model = load_nanosaur_transformer(dit_path, device=loading_device, dtype=dtype, requires_grad=False)
        if args.uncond_text_embedding:
            state = load_file(args.uncond_text_embedding, device="cpu")
            key = "uncond_gemma_embed"
            if key not in state:
                candidates = [name for name in state if name.startswith("uncond_gemma_embed")]
                if not candidates:
                    raise ValueError(f"No uncond_gemma_embed tensor found in {args.uncond_text_embedding}")
                key = candidates[0]
            self.uncond_text_embedding = state[key].to(device=accelerator.device, dtype=dtype)
        return model

    def compile_transformer(self, args, transformer):
        return transformer

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        cond = batch["gemma_embed"].to(device=accelerator.device, dtype=latents.dtype)
        if self.uncond_text_embedding is not None and self.cond_dropout > 0:
            drop_mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_dropout
            if drop_mask.any():
                cond = cond.clone()
                cond[drop_mask] = self.uncond_text_embedding.to(device=cond.device, dtype=cond.dtype)

        batch_size = latents.shape[0]
        t = sample_timesteps(batch_size, latents.device, latents.dtype, alpha=self.timestep_sampling_alpha)
        shape = [batch_size] + [1] * (latents.dim() - 1)
        t_view = t.view(shape)
        zt = (1 - t_view) * latents + t_view * noise

        zt = zt.to(device=accelerator.device, dtype=latents.dtype)
        cond = cond.to(device=accelerator.device, dtype=latents.dtype)
        t = t.to(device=accelerator.device, dtype=latents.dtype)

        if args.gradient_checkpointing:
            zt.requires_grad_(True)
            cond.requires_grad_(True)

        with accelerator.autocast():
            x0_pred, _ = transformer(zt, t, cond, return_x0=True)

        denom = (t + 0.05).view(shape)
        velocity = (zt - x0_pred) / denom
        target = (zt - latents) / denom
        return velocity.to(dtype=network_dtype), target.to(dtype=network_dtype)


def nanosaur_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, help="NanoSaur text encoder checkpoint path, used only for future sampling")
    parser.add_argument("--uncond_text_embedding", type=str, default=None, help="path to uncond_ns_te.safetensors for condition dropout")
    parser.add_argument("--timestep_sampling_alpha", type=float, default=2.0, help="NanoSaur timestep sampling alpha")
    parser.add_argument("--cond_dropout", type=float, default=0.1, help="NanoSaur condition dropout probability")
    return parser


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)
    parser = nanosaur_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    args.fp8_scaled = False

    trainer = NanoSaurNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
