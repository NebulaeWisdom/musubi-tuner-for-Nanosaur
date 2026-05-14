# NanoSaur LoRA Training with Musubi Tuner

本文从零开始说明如何在 `musubi-tuner` 中训练 NanoSaur-1.2B 的 LoRA。

当前 NanoSaur 支持是一个**最小可运行实现**：目标是先让 musubi trainer 能跑 NanoSaur LoRA 训练。它暂时不追求大规模训练优化，也暂不支持采样预览、block swap、fp8 base、gradient checkpointing 等高级功能。

---

## 1. 当前支持范围

已支持：

- NanoSaur diffusion model 加载；
- NanoSaur VAE latent cache；
- NanoSaur text encoder output cache；
- NanoSaur LoRA 注入；
- 使用 musubi `NetworkTrainer` 训练 LoRA；
- 使用 `accelerate launch` 单卡或多卡启动；
- 保存训练 LoRA；
- 同步导出 `-comfyui.safetensors` LoRA。

暂不支持：

- `--sample_prompts` 训练中采样；
- `--gradient_checkpointing`；
- `--blocks_to_swap`；
- `--fp8_base`；
- 全参训练；
- text encoder 训练。

如果你的目标是先跑起来，请按本文流程操作。

---

## 2. 目录关系

本文假设你在这个大仓库里工作：

```text
Nanosaur-1.2B-Train/
├─ download_model.py
├─ dataset/
├─ nanosaur_support/
│  ├─ nanosaur_diffusion_model.safetensors
│  ├─ nanosaur_text_encoder.safetensors
│  └─ nanosaur_vae_decoder.safetensors
└─ trainer/
   └─ musubi-tuner/
      ├─ src/musubi_tuner/
      │  ├─ nanosaur_cache_latents.py
      │  ├─ nanosaur_cache_text_encoder_outputs.py
      │  ├─ nanosaur_train_network.py
      │  ├─ nanosaur_utils.py
      │  └─ networks/lora_nanosaur.py
      ├─ examples/nanosaur_dataset.toml
      └─ examples/nanosaur_minimal_train.sh
```

推荐从 musubi-tuner 目录运行命令：

```bash
cd trainer/musubi-tuner
```

---

## 3. 环境准备

### 3.1 创建虚拟环境

不要在基础 Python 环境里直接安装包。建议在项目内创建 venv：

#### Linux / WSL / Git Bash

```bash
cd Nanosaur-1.2B-Train
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
cd Nanosaur-1.2B-Train
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.2 安装依赖

本项目根目录已有 `requirements.txt`，可先安装它：

```bash
pip install -r requirements.txt
```

然后进入 musubi-tuner，安装 musubi 自身：

```bash
cd trainer/musubi-tuner
pip install -e .
```

如果缺少 ModelScope 下载依赖：

```bash
pip install modelscope
```

如果后面运行提示缺少 `accelerate`，安装：

```bash
pip install accelerate
```

### 3.3 配置 accelerate

首次使用建议运行：

```bash
accelerate config
```

如果你只想先单卡跑通，也可以直接用：

```bash
accelerate launch --num_processes 1 ...
```

---

## 4. 下载 NanoSaur 模型

回到项目根目录：

```bash
cd Nanosaur-1.2B-Train
python download_model.py
```

该脚本会下载：

```text
nanosaur_support/nanosaur_diffusion_model.safetensors
nanosaur_support/nanosaur_text_encoder.safetensors
nanosaur_support/nanosaur_vae_decoder.safetensors
```

模型来源：

```text
xiaobaibai030/well9472-Nanosaur-1.2B-Preview
```

下载完成后检查：

```bash
ls nanosaur_support
```

至少应看到：

```text
nanosaur_diffusion_model.safetensors
nanosaur_text_encoder.safetensors
nanosaur_vae_decoder.safetensors
```

---

## 5. 准备数据集

### 5.1 数据格式

把训练图片和同名 caption 文本放到项目根目录的 `dataset/`：

```text
dataset/
├─ image001.png
├─ image001.txt
├─ image002.jpg
├─ image002.txt
├─ image003.webp
└─ image003.txt
```

要求：

- 图片和 `.txt` 文件同名；
- `.txt` 中写该图片的 prompt/caption；
- 一张图片对应一个 caption；
- 支持常见图片格式：`.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp` 等。

示例：

```text
dataset/cat001.png
dataset/cat001.txt
```

`cat001.txt` 内容可以是：

```text
a cute orange cat sitting on a wooden table, soft lighting, detailed fur
```

### 5.2 建议先小数据测试

第一次不要直接上大数据。建议：

```text
10 - 100 张图片先跑通
再扩大到 1K / 10K
```

---

## 6. Dataset config

musubi-tuner 使用 TOML 配置数据集。

已提供示例：

```text
trainer/musubi-tuner/examples/nanosaur_dataset.toml
```

内容：

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
image_directory = "../../../dataset"
cache_directory = "../../../cache/nanosaur_1024_fp16"
num_repeats = 1
```

如果你在 `trainer/musubi-tuner` 目录运行命令，上面的相对路径指向：

```text
../../../dataset
../../../cache/nanosaur_1024_fp16
```

也就是大仓库根目录下的：

```text
dataset/
cache/nanosaur_1024_fp16/
```

### 6.1 如果你的数据不在默认目录

复制一份配置：

```bash
cp examples/nanosaur_dataset.toml examples/my_nanosaur_dataset.toml
```

修改：

```toml
image_directory = "/path/to/your/dataset"
cache_directory = "/path/to/your/cache/nanosaur_1024_fp16"
```

---

## 7. 第一步：缓存 VAE latents

进入 musubi-tuner：

```bash
cd trainer/musubi-tuner
```

运行：

```bash
python -m musubi_tuner.nanosaur_cache_latents \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --vae ../../nanosaur_support/nanosaur_vae_decoder.safetensors \
  --vae_dtype fp16 \
  --batch_size 4 \
  --skip_existing
```

作用：

1. 读取 `dataset/` 图片；
2. resize / crop 到 1024；
3. 用 NanoSaur VAE 编码 latent；
4. 写入 `cache/nanosaur_1024_fp16/`。

输出文件类似：

```text
cache/nanosaur_1024_fp16/image001_1024x1024_ns.safetensors
cache/nanosaur_1024_fp16/image002_1024x1024_ns.safetensors
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--dataset_config` | 数据集 TOML 配置 |
| `--vae` | NanoSaur VAE 权重 |
| `--vae_dtype fp16` | 用 fp16 缓存 VAE latent |
| `--batch_size` | VAE 编码 batch size，显存不足就调小 |
| `--skip_existing` | 已存在 cache 就跳过，方便断点续跑 |

---

## 8. 第二步：缓存 text encoder outputs

运行：

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../../nanosaur_support/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype fp16 \
  --batch_size 8 \
  --skip_existing
```

作用：

1. 读取每张图片对应的 `.txt` caption；
2. 用 NanoSaur text encoder 编码；
3. 写入 text embedding cache；
4. 额外生成空文本 embedding，用于 condition dropout。

输出类似：

```text
cache/nanosaur_1024_fp16/image001_ns_te.safetensors
cache/nanosaur_1024_fp16/image002_ns_te.safetensors
cache/nanosaur_1024_fp16/uncond_ns_te.safetensors
```

训练时会读取：

```text
latents            -> batch["latents"]
gemma text embeds  -> batch["gemma_embed"]
uncond embed       -> --uncond_text_embedding
```

---

## 9. 第三步：启动 LoRA 训练

单卡最小训练命令：

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../../nanosaur_support/nanosaur_diffusion_model.safetensors \
  --vae ../../nanosaur_support/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 16 \
  --network_alpha 4 \
  --network_dropout 0 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 1000 \
  --save_every_n_steps 500 \
  --output_dir ../../outputs/nanosaur_musubi_lora \
  --output_name nanosaur_lora
```

训练输出目录：

```text
outputs/nanosaur_musubi_lora/
```

每次保存时会生成：

```text
nanosaur_lora-000000500.safetensors
nanosaur_lora-000000500-comfyui.safetensors
```

其中：

- 不带 `-comfyui` 的文件是训练格式；
- `-comfyui.safetensors` 是 NanoSaur ComfyUI 风格导出。

---

## 10. 一键最小脚本

已提供脚本：

```text
examples/nanosaur_minimal_train.sh
```

运行：

```bash
cd trainer/musubi-tuner
bash examples/nanosaur_minimal_train.sh
```

该脚本会依次执行：

1. latent cache；
2. text encoder cache；
3. LoRA training。

如果你的路径不同，可以用环境变量覆盖：

```bash
DATASET_CONFIG=examples/my_nanosaur_dataset.toml \
MODEL_DIR=../../nanosaur_support \
CACHE_DIR=../../cache/nanosaur_1024_fp16 \
OUTPUT_DIR=../../outputs/my_nanosaur_lora \
bash examples/nanosaur_minimal_train.sh
```

---

## 11. 多卡训练

把 `--num_processes` 改成 GPU 数量即可，例如 4 卡：

```bash
accelerate launch --num_processes 4 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../../nanosaur_support/nanosaur_diffusion_model.safetensors \
  --vae ../../nanosaur_support/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 16 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1000 \
  --save_every_n_steps 500 \
  --output_dir ../../outputs/nanosaur_musubi_lora \
  --output_name nanosaur_lora
```

global batch size 计算：

```text
global_batch_size = dataset batch_size × num_processes × gradient_accumulation_steps
```

例如：

```text
1 × 4 × 4 = 16
```

---

## 12. 续训 / 加载已有 LoRA

如果只是加载已有 LoRA 继续训练，可以使用 musubi 的：

```text
--network_weights path/to/lora.safetensors
```

示例：

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../../nanosaur_support/nanosaur_diffusion_model.safetensors \
  --vae ../../nanosaur_support/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_weights ../../outputs/nanosaur_musubi_lora/nanosaur_lora-000000500.safetensors \
  --network_dim 16 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 2000 \
  --save_every_n_steps 500 \
  --output_dir ../../outputs/nanosaur_musubi_lora_resume \
  --output_name nanosaur_lora_resume
```

注意：

- `--network_weights` 应使用训练格式 LoRA，不是 `-comfyui.safetensors`；
- 目前最小实现重点是加载 LoRA 权重继续训练；完整 optimizer/scheduler 状态恢复依赖 musubi/accelerate 的 state 机制，后续可继续加强。

---

## 13. 重要参数解释

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `--network_dim` | `16` | LoRA rank，越大参数越多 |
| `--network_alpha` | `4` | LoRA alpha，当前沿用原项目默认 |
| `--network_dropout` | `0` | LoRA dropout，先用 0 跑通 |
| `--learning_rate` | `1e-4` | LoRA 学习率 |
| `--mixed_precision` | `bf16` | 推荐 NVIDIA 新卡用 bf16 |
| `--sdpa` | 开启 | 使用 PyTorch SDPA attention 路径 |
| `--max_train_steps` | `1000` 起 | 先小步数测试 |
| `--save_every_n_steps` | `500` | 每多少 step 保存一次 |
| `--cond_dropout` | `0.1` | 随机使用空文本条件的概率 |
| `--timestep_sampling_alpha` | `2.0` | NanoSaur 当前 timestep sampling 参数 |

---

## 14. 当前 NanoSaur 最小实现限制

请先不要使用这些参数：

```text
--sample_prompts
--gradient_checkpointing
--blocks_to_swap
--fp8_base
```

原因：当前目标是先跑通 NanoSaur LoRA 训练，以上能力还没有接入 NanoSaur adapter。

如果你传了这些参数，脚本会直接报错或提示暂不支持。

---

## 15. 常见问题

### 15.1 `No training items found`

通常说明 cache 没生成，或 dataset config 路径不对。

检查：

```text
cache/nanosaur_1024_fp16/*.safetensors
```

至少应该同时有：

```text
*_1024x1024_ns.safetensors
*_ns_te.safetensors
```

### 15.2 找不到 `nanosaur_support`

NanoSaur musubi adapter 会导入：

```python
nanosaur_support.model
nanosaur_support.vae
```

请从 `Nanosaur-1.2B-Train` 仓库内运行，或把仓库根目录加入 `PYTHONPATH`。

如果你在 `trainer/musubi-tuner` 下运行，通常可以：

```bash
export PYTHONPATH=../../:$PYTHONPATH
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "..\..;$env:PYTHONPATH"
```

### 15.3 CUDA 显存不足

先调小：

```text
cache latent batch_size
训练 dataset batch_size
gradient_accumulation_steps
network_dim
```

最小测试建议：

```text
batch_size = 1
network_dim = 8 或 16
max_train_steps = 100
```

### 15.4 没有生成 `uncond_ns_te.safetensors`

重新运行 text cache，不要带 `--skip_uncond`：

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../../nanosaur_support/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype fp16 \
  --batch_size 8 \
  --skip_existing
```

### 15.5 `--sample_prompts` 报错

正常。当前 NanoSaur 最小实现暂不支持训练中 sample。

先不要传 `--sample_prompts`。

---

## 16. 推荐第一次 smoke test

1. 准备 10 张图片和 10 个 `.txt`；
2. 下载模型；
3. cache latents；
4. cache text encoder outputs；
5. 训练 100 step。

训练命令：

```bash
accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --dit ../../nanosaur_support/nanosaur_diffusion_model.safetensors \
  --vae ../../nanosaur_support/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 8 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 100 \
  --save_every_n_steps 50 \
  --output_dir ../../outputs/nanosaur_smoke_test \
  --output_name nanosaur_smoke
```

如果成功，输出目录中应出现：

```text
nanosaur_smoke-000000050.safetensors
nanosaur_smoke-000000050-comfyui.safetensors
```

---

## 17. 后续可改进方向

当前文档只覆盖最小训练。

后续可继续补：

- 训练中 sample；
- 更完整的 resume optimizer/scheduler 状态说明；
- gradient checkpointing；
- block swap；
- 更高效的大规模 cache；
- NanoSaur LoRA 推理加载命令；
- ComfyUI 节点验证流程。
