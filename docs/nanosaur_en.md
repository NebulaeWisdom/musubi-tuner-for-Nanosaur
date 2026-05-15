# NanoSaur LoRA Training with Musubi Tuner

This document explains how to train a LoRA for NanoSaur-1.2B in `musubi-tuner` from scratch.

NanoSaur support has been gradually improved since musubi-tuner v0.2.15, with gradient checkpointing now implemented.

---

## 1. Currently Supported

Supported:

- NanoSaur diffusion model loading;
- NanoSaur VAE latent cache;
- NanoSaur text encoder output cache;
- NanoSaur LoRA injection;
- LoRA training using musubi `NetworkTrainer`;
- Single or multi-GPU launch via `accelerate launch`;
- gradient checkpointing (`--gradient_checkpointing`);
- Saving trained LoRA;
- Synchronized export of `-comfyui.safetensors` LoRA.

Not yet supported:

- `--sample_prompts` sampling during training;
- `--blocks_to_swap`;
- `--fp8_base`;
- Full parameter training;
- Text encoder training.

If your goal is to get things running first, follow the workflow in this document.

---

## 2. Directory Structure

This document assumes you are working within this repository layout:

```text
<repo-root>/
├─ dataset/
├─ model/
│  ├─ nanosaur_diffusion_model.safetensors
│  ├─ nanosaur_text_encoder.safetensors
│  └─ nanosaur_vae_decoder.safetensors
└─ musubi-tuner/
   ├─ src/musubi_tuner/
   │  ├─ nanosaur/
   │  │  ├─ __init__.py
   │  │  ├─ model.py
   │  │  └─ vae.py
   │  ├─ nanosaur_cache_latents.py
   │  ├─ nanosaur_cache_text_encoder_outputs.py
   │  ├─ nanosaur_train_network.py
   │  ├─ nanosaur_utils.py
   │  └─ networks/lora_nanosaur.py
   ├─ examples/nanosaur_dataset.toml
   ├─ examples/nanosaur_minimal_train.sh
   ├─ ns_cache_vae_latent.bat
   ├─ ns_cache_te_latent.bat
   └─ ns_train_lora_model.bat
```

It is recommended to run commands from the musubi-tuner directory:

```bash
cd musubi-tuner
```

---

## 3. Environment Setup

### 3.1 Create a Virtual Environment

Do not install packages directly in the base Python environment. It is recommended to create a venv within the project:

#### Linux / WSL / Git Bash

```bash
cd <repo-root>
python -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
cd <repo-root>
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3.2 Install Dependencies

A `requirements.txt` already exists at the repo root; install it first:

```bash
pip install -r requirements.txt
```

Then enter musubi-tuner and install musubi itself:

```bash
cd musubi-tuner
pip install -e .
```

A local Windows smoke test discovered: `sentencepiece==0.2.1` may cause a native segfault when reading the SentencePiece tokenizer proto embedded in NanoSaur weights. It is recommended to use a verified version within the project venv:

```bash
pip install sentencepiece==0.1.99
```

Do not install these dependencies in the system/base Python environment.

If ModelScope download dependencies are missing:

```bash
pip install modelscope
```

If you later encounter a missing `accelerate` error, install:

```bash
pip install accelerate
```

### 3.3 Configure accelerate

For first-time use, it is recommended to run:

```bash
accelerate config
```

If you just want to run on a single GPU first, you can also use:

```bash
accelerate launch --num_processes 1 ...
```

---

## 4. Download the NanoSaur Model

Download the model files from either of the following addresses and place them in the `<repo-root>/model/` directory:

- **HuggingFace:** https://huggingface.co/well9472/Nanosaur-1.2B-Preview
- **ModelScope:** https://modelscope.cn/models/xiaobaibai030/well9472-Nanosaur-1.2B-Preview

Files to download:

```text
nanosaur_diffusion_model.safetensors
nanosaur_text_encoder.safetensors
nanosaur_vae_decoder.safetensors
```

> If you have `huggingface_hub`, you can also download via command line:
> ```bash
> pip install huggingface_hub
> huggingface-cli download well9472/Nanosaur-1.2B-Preview nanosaur_diffusion_model.safetensors nanosaur_text_encoder.safetensors nanosaur_vae_decoder.safetensors --local-dir model/
> ```

Check after downloading:

```bash
ls model
```

You should at least see:

```text
nanosaur_diffusion_model.safetensors
nanosaur_text_encoder.safetensors
nanosaur_vae_decoder.safetensors
```

---

## 5. Prepare the Dataset

### 5.1 Data Format

Place training images and caption text files with matching names into `dataset/` at the repo root:

```text
dataset/
├─ image001.png
├─ image001.txt
├─ image002.jpg
├─ image002.txt
├─ image003.webp
└─ image003.txt
```

Requirements:

- The image and `.txt` file must have the same name;
- The `.txt` file should contain the prompt/caption for that image;
- One image maps to one caption;
- Common image formats are supported: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, etc.

Example:

```text
dataset/cat001.png
dataset/cat001.txt
```

The content of `cat001.txt` could be:

```text
a cute orange cat sitting on a wooden table, soft lighting, detailed fur
```

### 5.2 Test with a Small Dataset First

Do not start with a large dataset right away. It is recommended to:

```text
Get things running with 10 - 100 images first
Then scale up to 1K / 10K
```

---

## 6. Dataset Config

musubi-tuner uses TOML for dataset configuration.

An example is provided:

```text
musubi-tuner/examples/nanosaur_dataset.toml
```

Content:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "../dataset"
cache_directory = "../cache/nanosaur_1024_fp16"
num_repeats = 1
```

Explanation:

- `batch_size = 1` is the training micro-batch. NanoSaur activation memory is very high at 1024 resolution; setting it to 4 directly is not recommended. When an effective batch of 4 is needed, use `--gradient_accumulation_steps 4` as a training parameter.
- `enable_bucket = true` performs bucketing by image aspect ratio; after bucketing, latent cache filenames include the bucket resolution.
- `bucket_no_upscale = false` is the default value, written explicitly here. It will resize/crop images to the bucket nearest the target area, which is generally better for LoRA training.
- NanoSaur's current bucket step comes from the spatial alignment constraint of VAE/DINO patch16, and buckets are generated in multiples of 16 by default. It is not recommended to hardcode a change to 64. 64 is merely a coarser bucketing strategy, not a native requirement.

If you run commands from the `musubi-tuner` directory, the relative paths above point to:

```text
../dataset
../cache/nanosaur_1024_fp16
```

Which correspond to the following at the top-level repo root:

```text
dataset/
cache/nanosaur_1024_fp16/
```

### 6.1 If Your Data Is Not in the Default Directory

Copy the config:

```bash
cp examples/nanosaur_dataset.toml examples/my_nanosaur_dataset.toml
```

Modify:

```toml
image_directory = "/path/to/your/dataset"
cache_directory = "/path/to/your/cache/nanosaur_1024_fp16"
```

If you modify `enable_bucket`, `resolution`, or `cache_directory`, regenerate the latent cache. If you change `cache_directory`, the text encoder cache and `uncond_ns_te.safetensors` must also be regenerated in the new directory.

---

## 7. Step 1: Cache VAE Latents

Enter musubi-tuner:

```bash
cd musubi-tuner
```

Run:

```bash
python -m musubi_tuner.nanosaur_cache_latents \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --vae_dtype fp16 \
  --batch_size 1 \
  --num_workers 1 \
  --skip_existing
```

What it does:

1. Reads images from `dataset/`;
2. Resizes/crops to the bucket resolution around 1024;
3. Encodes latents using NanoSaur VAE;
4. Writes to `cache/nanosaur_1024_fp16/`.

Output files look like:

```text
cache/nanosaur_1024_fp16/image001_1024x1024_ns.safetensors
cache/nanosaur_1024_fp16/image002_0768x1344_ns.safetensors
```

Parameter explanation:

| Parameter | Description |
|---|---|
| `--dataset_config` | Dataset TOML config |
| `--vae` | NanoSaur VAE weights |
| `--vae_dtype fp16` | Cache VAE latents in fp16 |
| `--batch_size` | VAE encoding batch size; reduce if VRAM is insufficient |
| `--num_workers` | DataLoader worker count; use 1 on Windows for stability |
| `--skip_existing` | Skip existing cache entries, convenient for resuming |

---

## 8. Step 2: Cache Text Encoder Outputs

Run:

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../model/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype bf16 \
  --batch_size 8 \
  --skip_existing
```

The text encoder cache does not depend on bucket resolution; however, training looks for the text cache in the same `cache_directory` specified in the dataset config, so after changing the cache directory you must still regenerate or copy the text cache.

What it does:

1. Reads the `.txt` caption for each image;
2. Encodes using the NanoSaur text encoder;
3. Writes text embedding cache;
4. Additionally generates an empty text embedding for condition dropout.

Output looks like:

```text
cache/nanosaur_1024_fp16/image001_ns_te.safetensors
cache/nanosaur_1024_fp16/image002_ns_te.safetensors
cache/nanosaur_1024_fp16/uncond_ns_te.safetensors
```

During training, the following are read:

```text
latents            -> batch["latents"]
gemma text embeds  -> batch["gemma_embed"]
uncond embed       -> --uncond_text_embedding
```

---

## 9. Step 3: Launch LoRA Training

Minimal single-GPU training command:

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../model/nanosaur_diffusion_model.safetensors \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 8 \
  --network_alpha 4 \
  --network_dropout 0 \
  --optimizer_type AdamW \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --sdpa \
  --max_train_epochs 10 \
  --save_every_n_epochs 1 \
  --output_dir ../outputs/nanosaur_adamw_10epoch_lora \
  --output_name nanosaur_adamw_10epoch_lora
```

Training output directory:

```text
outputs/nanosaur_adamw_10epoch_lora/
```

Each save generates:

```text
nanosaur_adamw_10epoch_lora-000000001.safetensors
nanosaur_adamw_10epoch_lora-000000001-comfyui.safetensors
```

Where:

- Files without `-comfyui` are in training format;
- `-comfyui.safetensors` is the NanoSaur ComfyUI-style export.

---

## 10. One-Click Minimal Script

A script is provided:

```text
examples/nanosaur_minimal_train.sh
```

Run:

```bash
cd musubi-tuner
bash examples/nanosaur_minimal_train.sh
```

This script executes in sequence:

1. latent cache;
2. text encoder cache;
3. LoRA training.

If your paths differ, you can override them with environment variables:

```bash
DATASET_CONFIG=examples/my_nanosaur_dataset.toml \
MODEL_DIR=../model \
CACHE_DIR=../cache/nanosaur_1024_fp16 \
OUTPUT_DIR=../outputs/my_nanosaur_lora \
bash examples/nanosaur_minimal_train.sh
```

On Windows, three simple bat examples are also provided, all assuming they are run from the `musubi-tuner` directory and use the project root `venv`:

```text
ns_cache_vae_latent.bat  # Generate VAE latent cache
ns_cache_te_latent.bat   # Generate text encoder cache and uncond_ns_te.safetensors
ns_train_lora_model.bat  # AdamW 1e-4 training for 10 epochs, saving once per epoch
```

These bat files only use relative paths; edit them directly as needed.

---

## 11. Multi-GPU Training

Simply change `--num_processes` to the number of GPUs. For example, 4 GPUs:

```bash
accelerate launch --num_processes 4 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../model/nanosaur_diffusion_model.safetensors \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 16 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --sdpa \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1000 \
  --save_every_n_steps 500 \
  --output_dir ../outputs/nanosaur_musubi_lora \
  --output_name nanosaur_lora
```

Global batch size calculation:

```text
global_batch_size = dataset batch_size × num_processes × gradient_accumulation_steps
```

For example:

```text
1 × 4 × 4 = 16
```

---

## 12. Resume / Load Existing LoRA

To simply load an existing LoRA and continue training, you can use musubi's:

```text
--network_weights path/to/lora.safetensors
```

Example:

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../model/nanosaur_diffusion_model.safetensors \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_weights ../outputs/nanosaur_musubi_lora/nanosaur_lora-000000500.safetensors \
  --network_dim 16 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 2000 \
  --save_every_n_steps 500 \
  --output_dir ../outputs/nanosaur_musubi_lora_resume \
  --output_name nanosaur_lora_resume
```

Notes:

- `--network_weights` should use the training-format LoRA, not `-comfyui.safetensors`;
- The current minimal implementation focuses on loading LoRA weights to continue training; full optimizer/scheduler state recovery relies on musubi/accelerate's state mechanism and can be further enhanced later.

---

## 13. Key Parameter Explanation

| Parameter | Recommended | Description |
|---|---:|---|
| `--network_dim` | `8` or `16` | LoRA rank; larger means more parameters. Start with 8 if VRAM is tight |
| `--network_alpha` | `4` | LoRA alpha, currently follows the original project default |
| `--network_dropout` | `0` | LoRA dropout; start with 0 to get things running |
| `--learning_rate` | `1e-4` | LoRA learning rate |
| `--mixed_precision` | `bf16` | bf16 recommended for newer NVIDIA cards |
| `--gradient_accumulation_steps` | `4` | Simulate effective batch 4 with micro-batch 1, significantly reduces VRAM |
| `--gradient_checkpointing` | On | Trades recomputation for VRAM, recommended to enable |
| `--sdpa` | On | Use PyTorch SDPA attention path |
| `--max_train_epochs` | `10` | Use when controlling training rounds by epoch |
| `--save_every_n_epochs` | `1` | Save once per epoch |
| `--max_train_steps` | `1000` and up | You can also control training by steps |
| `--save_every_n_steps` | `500` | Save every N steps |
| `--cond_dropout` | `0.1` | Probability of randomly using empty text condition |
| `--timestep_sampling_alpha` | `2.0` | NanoSaur's current timestep sampling parameter |

---

## 14. Current NanoSaur Minimal Implementation Limitations

Please do not use these parameters for now:

```text
--sample_prompts
--blocks_to_swap
--fp8_base
```

Reason: these capabilities have not yet been wired into the NanoSaur adapter.

If you pass these parameters, the script will either error or report they are not yet supported.

Supported optimization feature:

```text
--gradient_checkpointing
```

Gradient checkpointing reduces VRAM by recomputing 26 encoder blocks and 2 text refine blocks. When `--gradient_checkpointing` is passed, the trainer automatically sets the transformer to training mode and enables checkpointing. Note: this makes each training step slightly slower (forward must be re-run during backpropagation), but significantly reduces peak VRAM.

---

## 15. FAQ

### 15.1 `No training items found`

Usually means the cache wasn't generated, or the dataset config path is incorrect.

Check:

```text
cache/nanosaur_1024_fp16/*.safetensors
```

At minimum, both should be present:

```text
*_1024x1024_ns.safetensors
*_ns_te.safetensors
```

### 15.2 Model definition not found

The NanoSaur model definition is in the `musubi_tuner.nanosaur` package and can be used directly after installing with `pip install -e .`.

### 15.3 CUDA out of memory

First, verify that `batch_size` in the dataset config is the micro-batch, not the effective total batch. At 1024 resolution, NanoSaur latents are approximately `96 x 64 x 64`, and the DiT backpropagates on a 64x64 token grid. Even when only training LoRA, a large amount of activations must be retained. `batch_size = 4` roughly quadruples activation memory.

Recommended config:

```toml
batch_size = 1
```

In the training command, use:

```bash
--gradient_accumulation_steps 4
```

This way the effective batch is still 4, but VRAM is calculated based on micro-batch 1.

If still insufficient, further reduce:

```text
cache latent batch_size
training dataset batch_size
network_dim
resolution
```

Minimal test recommendation:

```text
batch_size = 1
network_dim = 8 or 16
max_train_steps = 100
```

When VRAM is still tight, add `--gradient_checkpointing` to further reduce activation memory.

### 15.4 `uncond_ns_te.safetensors` was not generated

Re-run text cache without `--skip_uncond`:

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../model/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype bf16 \
  --batch_size 8 \
  --skip_existing
```

### 15.5 `--sample_prompts` errors

Normal. The current NanoSaur minimal implementation does not support in-training sampling yet.

Do not pass `--sample_prompts` for now.

### 15.6 Text encoder cache segfault or Permission denied on Windows

Two independent issues were found in local Windows smoke testing:

- `sentencepiece==0.2.1` may experience a native-level segfault when reading the `spiece_model` embedded in NanoSaur text encoder safetensors;
- Old code used `NamedTemporaryFile` to write to disk and then had SentencePiece open it, which on Windows can trigger `Permission denied` because the temporary file is still held by the current process. Linux typically allows this file access pattern, but loading the serialized proto directly from memory is less dependent on OS file semantics.

Solution:

1. Use `sentencepiece==0.1.99` within the project venv to avoid the locally observed `0.2.1` native segfault;
2. The current code has been changed to `SentencePieceProcessor.LoadFromSerializedProto(...)`, no longer loading the tokenizer via temporary files, and works on both Windows and Linux.

Performance impact: the tokenizer is loaded only once during text encoder initialization; `LoadFromSerializedProto` avoids temporary file I/O, and the current implementation converts directly from `uint8` tensor to bytes, which does not affect training iteration performance.

### 15.7 Text cache is all NaN

In local RTX 5080 testing, `--text_encoder_dtype fp16` caused the Gemma text encoder to output NaN, and the save logic replaced NaN with 0, rendering text conditioning essentially ineffective.

Please use:

```bash
--text_encoder_dtype bf16
```

Or use `fp32`. Training itself is still recommended with `--mixed_precision bf16`.

### 15.8 `Unknown architecture: ns` when saving LoRA

This was a code issue caused by missing metadata registration. NanoSaur metadata has now been registered in `sai_model_spec.py`:

```text
modelspec.architecture = Nanosaur-1.2B/lora
ss_base_model_version = nanosaur
```

### 15.9 Backward reports `Found dtype Half but expected Float`

This is a backpropagation error caused by `model_pred` and `target` having mismatched dtypes in the NanoSaur trainer. The fix has been applied in `nanosaur_train_network.py` to unify both to `network_dtype` before computing loss.

### 15.10 Why is the bucket step size 16? Should I change it to 64?

NanoSaur's default bucket step size is 16, which comes from the spatial alignment requirement of the native VAE / DINO patch16: input width and height must be aligned to at least multiples of 16, and the VAE latent space is approximately `H/16 x W/16`. Therefore 16 is the minimum hard constraint.

It is not recommended to hardcode the default to 64. 64 is a multiple of 16 and is generally safe at the model level, but it is merely a coarser bucketing strategy that reduces the number of buckets and changes resize/crop granularity, potentially lowering resolution utilization for images with extreme aspect ratios. If 64 is genuinely needed, it should be implemented as an additional configurable option and validated as a multiple of 16.

The current example uses:

```toml
enable_bucket = true
bucket_no_upscale = false
```

`bucket_no_upscale = false` is the default value and is written explicitly to avoid misunderstanding.

### 15.11 Is the saved LoRA in bf16? Why do the files seem larger?

The current NanoSaur trainer uses `dit_dtype` as the save dtype when saving LoRA. Typically, when the command passes:

```bash
--mixed_precision bf16
```

And `--dit_dtype` is not explicitly overridden, `dit_dtype` will be set to bf16, and `lora_nanosaur.py` will cast floating-point LoRA tensors to bf16 before saving. Therefore, a single LoRA file should normally be bf16, not fp32.

If you feel the files or output directory are larger, the usual reasons are:

1. `--network_dim` is larger. LoRA file size is roughly linear with rank; rank 8 is about 2x rank 4, and rank 16 is about 4x rank 4;
2. Each save generates both training-format and ComfyUI-format files: `*.safetensors` and `*-comfyui.safetensors`;
3. `--save_every_n_epochs 1` saves once per epoch, and a final version is also saved at the end of training, so the entire output directory accumulates multiple sets of files;
4. `--network_alpha` does not noticeably affect file size; it mainly affects the LoRA scale.

You can check the dtype inside a LoRA file with this command:

```bash
python -c "from safetensors.torch import load_file; p='path/to/lora.safetensors'; sd=load_file(p, device='cpu'); print(sorted({str(v.dtype) for v in sd.values()})); print(len(sd))"
```

If the output includes `torch.bfloat16`, the file was saved in bf16. ComfyUI files will have additional alpha tensors, so the tensor count is higher than the training format — this is normal.

---

## 16. Recommended First Smoke Test

1. Prepare 10 images and 10 `.txt` files;
2. Download the model;
3. Cache latents;
4. Cache text encoder outputs;
5. Train for 100 steps.

Training command:

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../model/nanosaur_diffusion_model.safetensors \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 8 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --sdpa \
  --max_train_steps 100 \
  --save_every_n_steps 50 \
  --output_dir ../outputs/nanosaur_smoke_test \
  --output_name nanosaur_smoke
```

If successful, the output directory should contain:

```text
nanosaur_smoke-000000050.safetensors
nanosaur_smoke-000000050-comfyui.safetensors
```

### 16.1 Local Verification Record

A real smoke test was completed in the project venv at `<repo-root>`:

- GPU: RTX 5080 16GB;
- Python: project `venv`;
- PyTorch: `2.8.0+cu128`;
- `sentencepiece`: `0.1.99`;
- dataset: 97 images, 1 of which was missing a matching `.txt`; the current data loader still generates an empty caption cache for it;
- latent cache: 97 `*_ns.safetensors`;
- text cache: 97 `*_ns_te.safetensors` + `uncond_ns_te.safetensors`;
- training: `max_train_steps=1`, `network_dim=4`, `mixed_precision=bf16`, `optimizer_type=AdamW`;
- additional verification: `optimizer_type=Adafactor` completed 1 step of training and saved LoRA; optimizer/scheduler still uses musubi's general logic, but optional capabilities like `bitsandbytes`, `wandb`, `tensorboard` depend on whether the corresponding dependencies are installed in the environment;
- output directory: `outputs/nanosaur_smoke_lora/`.

Verified generated and loadable:

```text
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-comfyui.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-step00000001.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-step00000001-comfyui.safetensors
```

The training-format LoRA is approximately 7.4 MB, containing 266 LoRA tensors; the ComfyUI export is approximately 7.4 MB, containing 399 tensors, with metadata `modelspec.architecture` set to `Nanosaur-1.2B/lora`.

---

## 17. Future Improvement Directions

The current documentation only covers minimal training.

Subsequent additions may include:

- In-training sampling;
- More complete resume optimizer/scheduler state documentation;
- Block swap;
- fp8 base training;
- More efficient large-scale caching;
- NanoSaur LoRA inference loading commands;
- ComfyUI node verification workflow.
