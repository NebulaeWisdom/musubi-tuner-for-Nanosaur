from __future__ import annotations

import math
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from .model import NanoSaurTransformer2DModel
from .vae import NanoSaurVAE

ROOT_DIR = Path(__file__).resolve().parent
DIFFUSION_CHECKPOINT_PATH = ROOT_DIR / "nanosaur_diffusion_model.safetensors"
TEXT_CHECKPOINT_PATH = ROOT_DIR / "nanosaur_text_encoder.safetensors"
VAE_CHECKPOINT_PATH = ROOT_DIR / "nanosaur_vae_decoder.safetensors"

TEXT_MAX_LENGTH = 128
LATENT_SCALE = 2.3623
LATENT_SHIFT = 0.0179

VAE_LATENT_DIM = 96
TEXT_VOCAB_SIZE = 262144
TEXT_EMBED_DIM = 640
TEXT_INTERMEDIATE_SIZE = 2048
TEXT_LAYERS = 18
TEXT_ATTENTION_HEADS = 4
TEXT_KEY_VALUE_HEADS = 1
TEXT_HEAD_DIM = 256
TEXT_MAX_POSITION_EMBEDDINGS = 32768
TEXT_SLIDING_WINDOW = 512

MODEL_CHANNELS = 96
MODEL_HEADS = 16
MODEL_DIM = 1536
MODEL_DECODER_HIDDEN = 2048
MODEL_ENCODER_LAYERS = 26
MODEL_DECODER_LAYERS = 3
MODEL_TEXT_BLOCKS = 2
MODEL_PATCH = 1


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


class NanoSaurVAEWrapper:
    def __init__(self, checkpoint_path: str | Path = VAE_CHECKPOINT_PATH, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        state_dict = _clean_state_dict(load_safetensors_file(Path(checkpoint_path), device="cpu"))
        self.model = NanoSaurVAE(latent_dim=VAE_LATENT_DIM).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        self.model.eval()
        print(f"loaded VAE model from {checkpoint_path}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.dtype if self.dtype is not None else x.dtype)
        return (self.model.encode(x) + LATENT_SHIFT) / LATENT_SCALE

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z * LATENT_SCALE) - LATENT_SHIFT
        z = z.to(self.device, dtype=self.dtype if self.dtype is not None else z.dtype)
        return self.model.decode(z)


class NanoSaurSentencePieceTokenizer:
    def __init__(self, spiece_model: torch.Tensor, max_length: int = TEXT_MAX_LENGTH) -> None:
        self.max_length = max_length
        model_bytes = bytes(spiece_model.cpu().numpy().tolist())
        self.processor = spm.SentencePieceProcessor()
        with tempfile.NamedTemporaryFile(suffix=".model") as handle:
            handle.write(model_bytes)
            handle.flush()
            self.processor.Load(handle.name)
        self.bos_token_id = 2
        self.pad_token_id = 0

    def __call__(self, captions: list[str], device: torch.device | str) -> dict[str, torch.Tensor]:
        rows = []
        for caption in captions:
            ids = [self.bos_token_id] + self.processor.EncodeAsIds(caption)
            ids = ids[: self.max_length]
            ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
            rows.append(ids)
        input_ids = torch.tensor(rows, device=device, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_text_encoder(device: str | torch.device, dtype: torch.dtype, checkpoint_path: str | Path = TEXT_CHECKPOINT_PATH):
    checkpoint = load_safetensors_file(Path(checkpoint_path), device="cpu")
    weights = {key: value for key, value in checkpoint.items() if key != "spiece_model"}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    config = Gemma3TextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_EMBED_DIM,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=TEXT_LAYERS,
        num_attention_heads=TEXT_ATTENTION_HEADS,
        num_key_value_heads=TEXT_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-6,
        qkv_bias=False,
        attention_bias=False,
        sliding_window=TEXT_SLIDING_WINDOW,
        use_cache=False,
    )
    text_encoder = Gemma3ForCausalLM(config)
    text_encoder.load_state_dict(weights, strict=True)
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
    tokenizer = NanoSaurSentencePieceTokenizer(checkpoint["spiece_model"])
    print(f"loaded text encoder from {checkpoint_path}")
    return tokenizer, text_encoder


@torch.no_grad()
def encode_text(captions: list[str], tokenizer: NanoSaurSentencePieceTokenizer, text_encoder: nn.Module, device: str | torch.device) -> torch.Tensor:
    tokens = tokenizer(captions, device=device)
    outputs = text_encoder(**tokens, output_hidden_states=True)
    return outputs.hidden_states[-1].detach().cpu().half()


class LoraNanoSaurTransformer2DModel(NanoSaurTransformer2DModel):
    def __init__(self) -> None:
        super().__init__(
            in_channels=MODEL_CHANNELS,
            num_groups=MODEL_HEADS,
            hidden_size=MODEL_DIM,
            decoder_hidden_size=MODEL_DECODER_HIDDEN,
            num_encoder_blocks=MODEL_ENCODER_LAYERS,
            num_decoder_blocks=MODEL_DECODER_LAYERS,
            num_text_blocks=MODEL_TEXT_BLOCKS,
            patch_size=MODEL_PATCH,
            txt_embed_dim=TEXT_EMBED_DIM,
        )
        self.projector = nn.Module()
        self.projector.conv = nn.Conv2d(MODEL_DIM, MODEL_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x, timestep, context=None, **kwargs):
        if context is None:
            raise ValueError("NanoSaurTransformer2DModel requires text context.")
        return_x0 = bool(kwargs.pop("return_x0", False))
        x0 = self._forward(x, timestep, context, **kwargs)
        if return_x0:
            return x0, None
        return (x - x0) / timestep.view(-1, 1, 1, 1)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float, prefix: str = "") -> int:
    count = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (
            child_prefix.startswith(("blocks.", "text_refine_blocks."))
            or (child_prefix.startswith("dec_net.res_blocks.") and ".mlp." in child_prefix)
        ):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += inject_lora(child, rank, alpha, dropout, child_prefix)
    return count


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().contiguous() for name, param in model.named_parameters() if "lora_" in name}


def comfy_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"diffusion_model.{name}"
        state[f"{prefix}.lora_up.weight"] = module.lora_up.weight.detach().cpu().contiguous()
        state[f"{prefix}.lora_down.weight"] = module.lora_down.weight.detach().cpu().contiguous()
        state[f"{prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current_state = model.state_dict()
    lora_keys = {name for name in current_state if "lora_" in name}
    if set(state) != lora_keys:
        raise RuntimeError("LoRA checkpoint keys do not match the model LoRA keys.")
    current_state.update(state)
    model.load_state_dict(current_state, strict=True)


def build_lora_diffusion_model(
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    alpha: float,
    dropout: float,
    checkpoint_path: str | Path = DIFFUSION_CHECKPOINT_PATH,
) -> nn.Module:
    weights = _clean_state_dict(load_safetensors_file(Path(checkpoint_path), device="cpu"))
    model = LoraNanoSaurTransformer2DModel()
    model.load_state_dict(weights, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    lora_layers = inject_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    model = model.to(device=device, dtype=dtype)
    print(f"loaded diffusion model from {checkpoint_path} and injected LoRA into {lora_layers} linear layers")
    return model