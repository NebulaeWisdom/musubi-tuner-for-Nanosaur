from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .model_lora import NanoSaurVAEWrapper, build_lora_diffusion_model, comfy_lora_state_dict, load_lora_state_dict, lora_state_dict

ROOT_DIR = Path(__file__).resolve().parent
LATENT_CACHE_PATH = ROOT_DIR / "cache" / "lora_latents.pt"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints_lora"
SAMPLES_DIR = ROOT_DIR / "samples_lora"
RUNS_DIR = ROOT_DIR / "runs_lora"

LR = 1e-4
WARMUP_STEPS = 0
EPOCHS = 10000
MIXED_PRECISION = "bf16"
SAVE_EVERY = 500
COND_DROPOUT = 0.1
CLIP_GRAD = 1.0
LOG_WINDOW = 10
TIME_SAMPLING_ALPHA = 2.0
BATCH_SIZE = 1
NUM_WORKERS = 4
LORA_RANK = 16
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.0
SAMPLE_INTERVAL = 500
SAMPLE_STEPS = 40
SAMPLE_COUNT = 4
SAMPLE_CFG = 7.0
SAMPLE_SHIFT = 4.0

torch.set_float32_matmul_precision("high")


class CachedLoraDataset(Dataset):
    def __init__(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        self.latents = state["latents"].float()
        self.text_embeddings = state["text_embeddings"].float()
        self.uncond_text_embedding = state["uncond_text_embedding"].float()
        self.captions = list(state["captions"])
        self.keys = list(state["keys"])
        if self.latents.shape[0] != self.text_embeddings.shape[0]:
            raise ValueError("cache latents and text_embeddings have different lengths")

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "latent": self.latents[index],
            "text_embedding": self.text_embeddings[index],
            "caption": self.captions[index],
            "key": self.keys[index],
        }


class LossWindow:
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffers: dict[str, deque[float]] = {}

    def update(self, values: dict[str, torch.Tensor | float]) -> None:
        for name, value in values.items():
            numeric = float(value.detach().mean().item()) if torch.is_tensor(value) else float(value)
            self.buffers.setdefault(name, deque(maxlen=self.size)).append(numeric)

    def averages(self) -> dict[str, float]:
        return {name: sum(buffer) / len(buffer) for name, buffer in self.buffers.items()}


def sample_timesteps(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mu = torch.log(torch.tensor(TIME_SAMPLING_ALPHA, device=device, dtype=dtype))
    return torch.sigmoid(torch.randn((batch,), device=device, dtype=dtype) + mu)


def rectified_flow_loss(model: torch.nn.Module, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    batch = x.size(0)
    t = sample_timesteps(batch, x.device, x.dtype)
    shape = [batch] + [1] * (x.dim() - 1)
    z1 = torch.randn_like(x)
    zt = (1 - t.view(shape)) * x + t.view(shape) * z1
    model_output, _ = model(zt, t, cond, return_x0=True)
    t_clamped = (t + 0.05).view(shape)
    velocity = (zt - model_output) / t_clamped
    target = (zt - x) / t_clamped
    return ((target - velocity) ** 2).mean(dim=tuple(range(1, x.dim()))).mean()


def get_sampling_timesteps(steps: int, device: torch.device, dtype: torch.dtype, sample_shift: float | None = None) -> torch.Tensor:
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)[:-1]
    if sample_shift is not None and sample_shift > 0:
        timesteps = sample_shift * timesteps / (1 + (sample_shift - 1) * timesteps)
    return timesteps


def rectified_flow_sample(
    model: torch.nn.Module,
    z: torch.Tensor,
    cond: torch.Tensor,
    null_cond: torch.Tensor,
    steps: int,
    guidance_scale: float,
    sample_shift: float | None = None,
    path_drop_guidance: bool = True,
    cfg_start: float = 0.03,
    cfg_end: float = 0.8,
    use_momentum_guidance: bool = True,
    mg_alpha: float = 0.5,
    mg_beta: float = 0.6,
    show_progress: bool = False,
) -> torch.Tensor:
    latents = z.clone()
    batch = latents.size(0)
    device = latents.device
    dtype = latents.dtype
    latent_shape = [1] + [1] * (latents.dim() - 1)
    timesteps = get_sampling_timesteps(steps, device, dtype, sample_shift)
    momentum = None
    iterator = tqdm(range(steps), desc="sample", leave=False) if show_progress else range(steps)
    for index in iterator:
        t_curr = timesteps[index]
        t_next = timesteps[index + 1] if index + 1 < steps else torch.tensor(0.0, device=device, dtype=dtype)
        dt = t_curr - t_next
        t = t_curr.expand(batch)
        guided_output, _ = model(latents, t, cond, return_x0=True)
        guided = (latents - guided_output) / t_curr
        step_fraction = index / steps
        apply_cfg = cfg_start < step_fraction < cfg_end
        if null_cond is not None and apply_cfg:
            step_uncond = path_drop_guidance and index % 2 == 1
            unguided_output, _ = model(latents, t, null_cond, uncond=step_uncond, return_x0=True)
            unguided = (latents - unguided_output) / t_curr
            guided = unguided + guidance_scale * (guided - unguided)
            if use_momentum_guidance:
                if momentum is None:
                    momentum = guided.clone()
                effective_velocity = guided + mg_alpha * (guided - momentum)
                momentum = (1.0 - mg_beta) * guided + mg_beta * momentum
                guided = effective_velocity
        latents = latents - dt.view(latent_shape) * guided
    return latents


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, optimizer: Optimizer, scheduler, step: int) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    comfy_lora_path = CHECKPOINTS_DIR / f"lora_comfyui_{step:08d}.safetensors"
    training_lora_path = CHECKPOINTS_DIR / f"lora_training_resume_{step:08d}.safetensors"
    save_safetensors_file(comfy_lora_state_dict(unwrapped), comfy_lora_path)
    save_safetensors_file(lora_state_dict(unwrapped), training_lora_path)
    accelerator.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "comfy_lora_path": str(comfy_lora_path),
            "training_lora_path": str(training_lora_path),
        },
        CHECKPOINTS_DIR / "training_state.pt",
    )


def load_checkpoint(accelerator: Accelerator, model: torch.nn.Module, optimizer: Optimizer, scheduler, resume: bool) -> int:
    state_path = CHECKPOINTS_DIR / "training_state.pt"
    if not resume or not state_path.exists():
        return 0
    state = torch.load(state_path, map_location="cpu")
    lora_path = Path(state.get("training_lora_path") or state.get("lora_path") or state["comfy_lora_path"])
    load_lora_state_dict(accelerator.unwrap_model(model), load_safetensors_file(lora_path, device="cpu"))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"loaded LoRA checkpoint {lora_path}")
    return int(state["step"]) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters for the nanosaur DeCo diffusion model.")
    parser.add_argument("--cache-path", type=Path, default=LATENT_CACHE_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dataset = CachedLoraDataset(args.cache_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False)
    vae = NanoSaurVAEWrapper(device=accelerator.device, dtype=dtype if torch.cuda.is_available() else None)
    model = build_lora_diffusion_model(accelerator.device, dtype, args.rank, args.alpha, args.dropout)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, args.epochs * len(loader))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, args.resume)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))) if accelerator.is_main_process else None
    loss_window = LossWindow(LOG_WINDOW)
    fixed = [dataset[index] for index in range(min(SAMPLE_COUNT, len(dataset)))]
    fixed_latents = torch.stack([row["latent"] for row in fixed]).to(accelerator.device, dtype=dtype)
    fixed_cond = torch.stack([row["text_embedding"] for row in fixed]).to(accelerator.device, dtype=dtype)
    fixed_noise = torch.randn_like(fixed_latents)
    uncond = dataset.uncond_text_embedding.to(accelerator.device, dtype=dtype).unsqueeze(0)

    step = start_step
    for epoch in range(args.epochs):
        progress = tqdm(loader, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for batch in progress:
            latents = batch["latent"].to(accelerator.device, dtype=dtype)
            cond = batch["text_embedding"].to(accelerator.device, dtype=dtype)
            mask = torch.rand(cond.size(0), device=cond.device) < COND_DROPOUT
            if mask.any():
                cond = cond.clone()
                cond[mask] = uncond.expand(cond.size(0), -1, -1)[mask]
            optimizer.zero_grad(set_to_none=True)
            mse = rectified_flow_loss(model, latents, cond)
            loss = mse
            accelerator.backward(loss)
            accelerator.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], CLIP_GRAD)
            optimizer.step()
            scheduler.step()
            metrics = {"loss": loss, "mse": mse}
            if accelerator.is_main_process:
                loss_window.update(metrics)
                avgs = loss_window.averages()
                avgs["lr"] = optimizer.param_groups[0]["lr"]
                postfix = {name: f"{value:.4f}" for name, value in avgs.items() if name != "lr"}
                postfix["lr"] = f"{avgs['lr']:.3e}"
                progress.set_postfix(postfix)
                if writer is not None and step % 10 == 0:
                    for name, value in avgs.items():
                        writer.add_scalar(f"train/{name}", value, step)
                if step > 0 and step % SAVE_EVERY == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, step)
                if step % SAMPLE_INTERVAL == 0:
                    with torch.no_grad():
                        sampled = rectified_flow_sample(
                            model,
                            fixed_noise,
                            fixed_cond,
                            uncond.expand(fixed_cond.size(0), -1, -1),
                            SAMPLE_STEPS,
                            SAMPLE_CFG,
                            SAMPLE_SHIFT,
                            show_progress=accelerator.is_main_process,
                        )
                        decoded = vae.decode(sampled).to(accelerator.device)
                        target = vae.decode(fixed_latents).to(accelerator.device)
                        grid = make_grid(((torch.cat([target, decoded], dim=0) + 1.0) / 2.0).clamp(0, 1), nrow=len(fixed)).cpu()
                        save_image(grid, SAMPLES_DIR / f"sample_{step}.png")
                        if writer is not None:
                            writer.add_image("samples", grid, step)
            step += 1
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, scheduler, step)


if __name__ == "__main__":
    main()