from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from .model_lora import TEXT_MAX_LENGTH, NanoSaurVAEWrapper, build_text_encoder, encode_text

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = ROOT_DIR / "cache" / "lora_latents.pt"
IMAGE_SIZE = 1024
BATCH_SIZE = 8
NUM_WORKERS = 8


class ImageTextFolderDataset(Dataset):
    def __init__(self, dataset_dir: Path, image_size: int) -> None:
        self.dataset_dir = dataset_dir
        self.items = []
        for image_path in sorted(dataset_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                self.items.append((image_path, caption_path))
        if not self.items:
            raise ValueError(f"no image/.txt pairs found in {dataset_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, caption_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        caption = caption_path.read_text(encoding="utf-8").strip()
        return {
            "image": self.transform(image),
            "caption": caption,
            "key": image_path.stem,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache PS-VAE latents and Gemma text embeddings for LoRA training.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Folder with image files and same-name .txt captions.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="Output cache .pt file.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Square training resolution; must be divisible by 16.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dataset = ImageTextFolderDataset(args.dataset_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device == "cuda")

    vae = NanoSaurVAEWrapper(device=device, dtype=dtype)
    tokenizer, text_encoder = build_text_encoder(device, dtype)

    latents: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    captions: list[str] = []
    keys: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache_lora"):
            latents.append(vae.encode(batch["image"]).cpu().half())
            text_embeddings.append(encode_text(list(batch["caption"]), tokenizer, text_encoder, device))
            captions.extend(batch["caption"])
            keys.extend(batch["key"])

    uncond_text_embedding = encode_text([""], tokenizer, text_encoder, device)[0]
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.cat(latents, dim=0),
            "text_embeddings": torch.cat(text_embeddings, dim=0),
            "uncond_text_embedding": uncond_text_embedding,
            "captions": captions,
            "keys": keys,
            "image_size": args.image_size,
            "text_max_length": TEXT_MAX_LENGTH,
        },
        args.cache_path,
    )
    print(f"saved {len(dataset)} cached samples to {args.cache_path}")


if __name__ == "__main__":
    main()