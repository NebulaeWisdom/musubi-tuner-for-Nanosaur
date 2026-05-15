import math
from pathlib import Path

from einops import rearrange
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file as load_safetensors_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from .nanosaur import NanoSaurTransformer2DModel, NanoSaurVAE


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


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


class NanoSaurVAEWrapper:
    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        state_dict = clean_state_dict(load_safetensors_file(Path(checkpoint_path), device="cpu"))
        self.model = NanoSaurVAE(latent_dim=VAE_LATENT_DIM).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        self.model.eval()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)
        return self

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
        if spiece_model.dtype != torch.uint8 or spiece_model.dim() != 1:
            raise ValueError(f"spiece_model must be a 1D torch.uint8 tensor, got shape={tuple(spiece_model.shape)}, dtype={spiece_model.dtype}")
        model_bytes = spiece_model.detach().cpu().contiguous().numpy().tobytes()
        self.processor = spm.SentencePieceProcessor()
        self.processor.LoadFromSerializedProto(model_bytes)
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


def build_text_encoder(device: str | torch.device, dtype: torch.dtype, checkpoint_path: str | Path):
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
    return tokenizer, text_encoder


@torch.no_grad()
def encode_text(captions: list[str], tokenizer: NanoSaurSentencePieceTokenizer, text_encoder: nn.Module, device: str | torch.device) -> torch.Tensor:
    tokens = tokenizer(captions, device=device)
    outputs = text_encoder(**tokens, output_hidden_states=True)
    return outputs.hidden_states[-1].detach().cpu().half()


class NanoSaurTransformerForTraining(NanoSaurTransformer2DModel):
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
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def forward(self, x, timestep, context=None, **kwargs):
        if context is None:
            raise ValueError("NanoSaurTransformer2DModel requires text context.")
        return_x0 = bool(kwargs.pop("return_x0", False))
        x0 = self._forward(x, timestep, context, **kwargs)
        if return_x0:
            return x0, None
        return (x - x0) / timestep.view(-1, 1, 1, 1)

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def _use_checkpoint(self) -> bool:
        return self.training and self.gradient_checkpointing

    def _checkpoint(self, func, *args, **kwargs):
        """Wrap a block call with gradient checkpointing when enabled."""
        if self._use_checkpoint():
            if self.cpu_offload_checkpointing:
                from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
                device = next(func.parameters()).device
                func = create_cpu_offloading_wrapper(func, device)
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    def _forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, uncond: bool = False, token_weights: torch.Tensor | None = None) -> torch.Tensor:
        device = self.s_embedder.proj.weight.device
        embed_dtype = self.s_embedder.proj.weight.dtype
        if context.device != device:
            context = context.to(device)
        y_emb = self.y_embedder(context).view(context.size(0), -1, self.hidden_size).to(embed_dtype)
        batch, _, height, width = x.shape
        x_tokens = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        xpos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)
        t_emb = self.t_embedder(timesteps.view(-1)).view(batch, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t_emb)
        y_latent = y_emb.to(dtype=t_emb.dtype)

        for block in self.text_refine_blocks:
            y_latent = self._checkpoint(block, y_latent, condition)

        s = self.s_embedder(x_tokens)
        h_patches, w_patches = height // self.patch_size, width // self.patch_size

        # ---- Encoder forward section (sprint_num_f) ----
        for i in range(self.sprint_num_f):
            s = self._checkpoint(
                self.blocks[i], s, y_latent, condition, xpos,
                shared_ada_ln=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, h=h_patches, w=w_patches,
                y_token_weights=token_weights,
            )

        s_enc = s
        s_sparse = s
        # ---- Sprint middle section (sprint_num_g) ----
        for i in range(self.sprint_num_f, self.sprint_num_f + self.sprint_num_g):
            if not uncond:
                s_sparse = self._checkpoint(
                    self.blocks[i], s_sparse, y_latent, condition, xpos,
                    shared_ada_ln=self.shared_encoder_adaLN,
                    y_token_weights=token_weights,
                )

        g_pad = self.mask_token2.expand_as(s_sparse) if uncond else s_sparse
        s = self._sprint_fuse(s_enc, g_pad)

        # ---- Encoder backward section (sprint_num_h) ----
        for i in range(self.sprint_num_f + self.sprint_num_g, len(self.blocks)):
            s = self._checkpoint(
                self.blocks[i], s, y_latent, condition, xpos,
                shared_ada_ln=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, h=h_patches, w=w_patches,
                y_token_weights=token_weights,
            )

        s = torch.nn.functional.silu(t_emb + s)
        batch_size, length, _ = s.shape
        x_dec = x_tokens.reshape(batch_size * length, self.in_channels, self.patch_size**2).transpose(1, 2)
        s_dec = s.view(batch_size * length, self.hidden_size)
        x_dec = self.x_embedder(x_dec)
        x_dec = self.dec_net(x_dec, s_dec).transpose(1, 2).reshape(batch_size, length, -1)

        return rearrange(
            x_dec,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=height // self.patch_size,
            w=width // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels,
        )


def load_nanosaur_transformer(
    checkpoint_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> NanoSaurTransformerForTraining:
    weights = clean_state_dict(load_safetensors_file(Path(checkpoint_path), device="cpu"))
    model = NanoSaurTransformerForTraining()
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(requires_grad)
    model.eval()
    model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def sample_timesteps(batch: int, device: torch.device, dtype: torch.dtype, alpha: float = 2.0) -> torch.Tensor:
    mu = torch.log(torch.tensor(alpha, device=device, dtype=dtype))
    return torch.sigmoid(torch.randn((batch,), device=device, dtype=dtype) + mu)
