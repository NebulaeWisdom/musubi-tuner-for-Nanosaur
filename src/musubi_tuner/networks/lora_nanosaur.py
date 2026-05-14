import math
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from musubi_tuner.utils import model_utils


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


def _is_nanosaur_lora_target(child_prefix: str) -> bool:
    return child_prefix.startswith(("blocks.", "text_refine_blocks.")) or (
        child_prefix.startswith("dec_net.res_blocks.") and ".mlp." in child_prefix
    )


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float, prefix: str = "") -> int:
    count = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _is_nanosaur_lora_target(child_prefix):
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
        missing = sorted(lora_keys - set(state))[:10]
        extra = sorted(set(state) - lora_keys)[:10]
        raise RuntimeError(f"LoRA checkpoint keys do not match. Missing={missing}, extra={extra}")
    current_state.update(state)
    model.load_state_dict(current_state, strict=True)


class NanoSaurLoRANetwork(nn.Module):
    def __init__(self, transformer: nn.Module, multiplier: float, network_dim: int, network_alpha: float, dropout: float) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.network_dim = int(network_dim)
        self.network_alpha = float(network_alpha)
        self.dropout = float(dropout)
        # Keep a reference without registering the whole base transformer as a child
        # module. The transformer is prepared separately by musubi/accelerate; this
        # network object only exposes LoRA parameters to the optimizer/checkpoint hooks.
        object.__setattr__(self, "transformer", transformer)
        self.lora_layer_count = inject_lora(transformer, self.network_dim, self.network_alpha, self.dropout)
        if self.lora_layer_count == 0:
            raise RuntimeError("No NanoSaur LoRA target layers were found.")

    def apply_to(self, text_encoders, unet, apply_text_encoder: bool = False, apply_unet: bool = True):
        return

    def prepare_grad_etc(self, unet):
        self.train()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.transformer.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        params = [param for param in self.get_trainable_params() if param.requires_grad]
        return [{"params": params, "lr": unet_lr}], ["nanosaur_lora"]

    def enable_gradient_checkpointing(self):
        return

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        return

    def get_trainable_params(self):
        return [param for name, param in self.transformer.named_parameters() if "lora_" in name]

    def parameters(self, recurse: bool = True):
        return iter(self.get_trainable_params())

    def state_dict(self, *args, **kwargs):
        return lora_state_dict(self.transformer)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        load_lora_state_dict(self.transformer, state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def load_weights(self, file):
        state = load_file(file, device="cpu")
        load_lora_state_dict(self.transformer, state)
        return f"loaded {len(state)} NanoSaur LoRA tensors"

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None
        state = lora_state_dict(self.transformer)
        if dtype is not None:
            state = {key: value.to(dtype=dtype) for key, value in state.items()}
        metadata = {} if metadata is None else dict(metadata)
        metadata.update(
            {
                "ss_network_module": "musubi_tuner.networks.lora_nanosaur",
                "ss_network_dim": str(self.network_dim),
                "ss_network_alpha": str(self.network_alpha),
                "ss_network_dropout": str(self.dropout),
                "ss_architecture": "nanosaur",
            }
        )

        if os.path.splitext(str(file))[1] == ".safetensors":
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state, file, metadata)

            comfy_file = str(file).removesuffix(".safetensors") + "-comfyui.safetensors"
            comfy_state = comfy_lora_state_dict(self.transformer)
            if dtype is not None:
                comfy_state = {key: value.to(dtype=dtype) if torch.is_floating_point(value) else value for key, value in comfy_state.items()}
            comfy_metadata = dict(metadata)
            comfy_metadata["ss_format"] = "nanosaur_comfyui"
            save_file(comfy_state, comfy_file, comfy_metadata)
        else:
            torch.save(state, file)

    def apply_max_norm_regularization(self, max_norm_value, device):
        return 0, 0.0, 0.0

    def merge_to(self, text_encoders, unet, weights_sd, dtype, device):
        raise NotImplementedError("Merging NanoSaur LoRA weights is not implemented in the minimal adapter.")


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders,
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        raise ValueError("network_dim is required for NanoSaur LoRA")
    if network_alpha is None:
        network_alpha = network_dim
    dropout = 0.0 if neuron_dropout is None else float(neuron_dropout)
    return NanoSaurLoRANetwork(unet, multiplier, int(network_dim), float(network_alpha), dropout)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders=None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    if unet is None:
        raise ValueError("unet is required to create NanoSaur LoRA from weights")
    rank = None
    for key, value in weights_sd.items():
        if ".lora_down.weight" in key or key.endswith("lora_down.weight"):
            rank = value.shape[0]
            break
    if rank is None:
        raise ValueError("Could not infer NanoSaur LoRA rank from weights")
    network = NanoSaurLoRANetwork(unet, multiplier, int(rank), float(rank), 0.0)
    load_lora_state_dict(unet, weights_sd)
    return network
