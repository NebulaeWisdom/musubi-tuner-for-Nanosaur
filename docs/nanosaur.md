# NanoSaur LoRA Training with Musubi Tuner

本文从零开始说明如何在 `musubi-tuner` 中训练 NanoSaur-1.2B 的 LoRA。

当前 NanoSaur 支持是一个**最小可运行实现**：目标是先让 musubi trainer 能跑 NanoSaur LoRA 训练。它暂时不追求大规模训练优化，也暂不支持采样预览、block swap、fp8 base、gradient checkpointing 等高级功能。原生 NanoSaur LoRA 训练脚本同样没有实现 gradient checkpointing，省显存优先使用小 micro-batch 加梯度累积。

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
<repo-root>/
├─ download_model.py
├─ dataset/
├─ model/
│  ├─ nanosaur_diffusion_model.safetensors
│  ├─ nanosaur_text_encoder.safetensors
│  └─ nanosaur_vae_decoder.safetensors
├─ nanosaur_support/
└─ musubi-tuner/
   ├─ src/musubi_tuner/
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

推荐从 musubi-tuner 目录运行命令：

```bash
cd musubi-tuner
```

---

## 3. 环境准备

### 3.1 创建虚拟环境

不要在基础 Python 环境里直接安装包。建议在项目内创建 venv：

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

### 3.2 安装依赖

本项目根目录已有 `requirements.txt`，可先安装它：

```bash
pip install -r requirements.txt
```

然后进入 musubi-tuner，安装 musubi 自身：

```bash
cd musubi-tuner
pip install -e .
```

Windows 本地 smoke test 发现：`sentencepiece==0.2.1` 在读取 NanoSaur 权重内置的 SentencePiece tokenizer proto 时可能直接段错误；建议在项目 venv 内使用已验证的版本：

```bash
pip install sentencepiece==0.1.99
```

注意不要在系统/base Python 环境安装这些依赖。

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
cd <repo-root>
python download_model.py
```

该脚本会下载：

```text
model/nanosaur_diffusion_model.safetensors
model/nanosaur_text_encoder.safetensors
model/nanosaur_vae_decoder.safetensors
```

模型来源：

```text
xiaobaibai030/well9472-Nanosaur-1.2B-Preview
```

下载完成后检查：

```bash
ls model
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
musubi-tuner/examples/nanosaur_dataset.toml
```

内容：

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

说明：

- `batch_size = 1` 是训练 micro-batch。1024 分辨率下 NanoSaur 激活显存很高，不建议直接设成 4；需要等效 batch 4 时，用训练参数 `--gradient_accumulation_steps 4`。
- `enable_bucket = true` 会按图片长宽比分桶；分桶后 latent cache 文件名包含 bucket resolution。
- `bucket_no_upscale = false` 是默认值，这里显式写出。它会把图片 resize/crop 到目标面积附近的 bucket，通常更适合 LoRA 训练。
- NanoSaur 当前 bucket 步长来自 VAE/DINO patch16 的空间对齐约束，默认按 16 的倍数生成 bucket；不建议硬改为 64。64 只是更粗的分桶策略，不是原生要求。

如果你在 `musubi-tuner` 目录运行命令，上面的相对路径指向：

```text
../dataset
../cache/nanosaur_1024_fp16
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

如果修改了 `enable_bucket`、`resolution` 或 `cache_directory`，请重新生成 latent cache；如果换了 `cache_directory`，text encoder cache 和 `uncond_ns_te.safetensors` 也需要在新目录重新生成。

---

## 7. 第一步：缓存 VAE latents

进入 musubi-tuner：

```bash
cd musubi-tuner
```

运行：

```bash
python -m musubi_tuner.nanosaur_cache_latents \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --vae_dtype fp16 \
  --batch_size 1 \
  --num_workers 1 \
  --skip_existing
```

作用：

1. 读取 `dataset/` 图片；
2. resize / crop 到 1024 附近的 bucket resolution；
3. 用 NanoSaur VAE 编码 latent；
4. 写入 `cache/nanosaur_1024_fp16/`。

输出文件类似：

```text
cache/nanosaur_1024_fp16/image001_1024x1024_ns.safetensors
cache/nanosaur_1024_fp16/image002_0768x1344_ns.safetensors
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--dataset_config` | 数据集 TOML 配置 |
| `--vae` | NanoSaur VAE 权重 |
| `--vae_dtype fp16` | 用 fp16 缓存 VAE latent |
| `--batch_size` | VAE 编码 batch size，显存不足就调小 |
| `--num_workers` | DataLoader worker 数，Windows 上先用 1 更稳 |
| `--skip_existing` | 已存在 cache 就跳过，方便断点续跑 |

---

## 8. 第二步：缓存 text encoder outputs

运行：

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../model/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype bf16 \
  --batch_size 8 \
  --skip_existing
```

text encoder cache 不依赖 bucket resolution；但训练会在 dataset config 的同一个 `cache_directory` 查找 text cache，所以换 cache 目录后仍要重新生成或复制 text cache。

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
  --sdpa \
  --max_train_epochs 10 \
  --save_every_n_epochs 1 \
  --output_dir ../outputs/nanosaur_adamw_10epoch_lora \
  --output_name nanosaur_adamw_10epoch_lora
```

训练输出目录：

```text
outputs/nanosaur_adamw_10epoch_lora/
```

每次保存时会生成：

```text
nanosaur_adamw_10epoch_lora-000000001.safetensors
nanosaur_adamw_10epoch_lora-000000001-comfyui.safetensors
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
cd musubi-tuner
bash examples/nanosaur_minimal_train.sh
```

该脚本会依次执行：

1. latent cache；
2. text encoder cache；
3. LoRA training。

如果你的路径不同，可以用环境变量覆盖：

```bash
DATASET_CONFIG=examples/my_nanosaur_dataset.toml \
MODEL_DIR=../model \
CACHE_DIR=../cache/nanosaur_1024_fp16 \
OUTPUT_DIR=../outputs/my_nanosaur_lora \
bash examples/nanosaur_minimal_train.sh
```

Windows 下也提供了三个简单 bat 范例，均假设从 `musubi-tuner` 目录运行，并使用项目根目录的 `venv`：

```text
ns_cache_vae_latent.bat  # 生成 VAE latent cache
ns_cache_te_latent.bat   # 生成 text encoder cache 和 uncond_ns_te.safetensors
ns_train_lora_model.bat  # AdamW 1e-4 训练 10 epoch，每个 epoch 保存一次
```

这些 bat 只使用相对路径，按需直接编辑即可。

---

## 11. 多卡训练

把 `--num_processes` 改成 GPU 数量即可，例如 4 卡：

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
  --sdpa \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1000 \
  --save_every_n_steps 500 \
  --output_dir ../outputs/nanosaur_musubi_lora \
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

注意：

- `--network_weights` 应使用训练格式 LoRA，不是 `-comfyui.safetensors`；
- 目前最小实现重点是加载 LoRA 权重继续训练；完整 optimizer/scheduler 状态恢复依赖 musubi/accelerate 的 state 机制，后续可继续加强。

---

## 13. 重要参数解释

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `--network_dim` | `8` 或 `16` | LoRA rank，越大参数越多；显存紧张先用 8 |
| `--network_alpha` | `4` | LoRA alpha，当前沿用原项目默认 |
| `--network_dropout` | `0` | LoRA dropout，先用 0 跑通 |
| `--learning_rate` | `1e-4` | LoRA 学习率 |
| `--mixed_precision` | `bf16` | 推荐 NVIDIA 新卡用 bf16 |
| `--gradient_accumulation_steps` | `4` | 用 micro-batch 1 模拟等效 batch 4，显著降低显存 |
| `--sdpa` | 开启 | 使用 PyTorch SDPA attention 路径 |
| `--max_train_epochs` | `10` | 以 epoch 控制训练轮数时使用 |
| `--save_every_n_epochs` | `1` | 每个 epoch 保存一次 |
| `--max_train_steps` | `1000` 起 | 也可以用 step 控制训练 |
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

原生 `nanosaur_support/train_lora.py` 也没有实现 gradient checkpointing、block swap 或 activation offload。它默认显存较低的主要原因是 `BATCH_SIZE = 1`。在 musubi 中建议保持 dataset `batch_size = 1`，用 `--gradient_accumulation_steps` 提升等效 global batch。

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

当前代码会在常规仓库布局下自动把 `<repo-root>` 仓库根目录加入导入路径。若你移动了 `musubi-tuner` 目录，或仍然遇到该错误，请手动把仓库根目录加入 `PYTHONPATH`。

如果你在 `musubi-tuner` 下运行，通常可以：

```bash
export PYTHONPATH=..:$PYTHONPATH
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "..;$env:PYTHONPATH"
```

### 15.3 CUDA 显存不足

优先确认 dataset config 中的 `batch_size` 是 micro-batch，而不是等效总 batch。1024 分辨率下 NanoSaur latent 约为 `96 x 64 x 64`，DiT 会在 64x64 token 网格上反传；即使只训练 LoRA，也需要保留大量激活。`batch_size = 4` 会把激活显存近似放大 4 倍。

推荐配置：

```toml
batch_size = 1
```

训练命令中使用：

```bash
--gradient_accumulation_steps 4
```

这样等效 batch 仍是 4，但显存按 micro-batch 1 计算。

如果仍然不足，再调小：

```text
cache latent batch_size
训练 dataset batch_size
network_dim
resolution
```

最小测试建议：

```text
batch_size = 1
network_dim = 8 或 16
max_train_steps = 100
```

注意：当前 NanoSaur musubi adapter 和原生 NanoSaur LoRA 脚本都没有实现 gradient checkpointing；传 `--gradient_checkpointing` 会直接报错。

### 15.4 没有生成 `uncond_ns_te.safetensors`

重新运行 text cache，不要带 `--skip_uncond`：

```bash
python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config ./examples/nanosaur_dataset.toml \
  --text_encoder ../model/nanosaur_text_encoder.safetensors \
  --text_encoder_dtype bf16 \
  --batch_size 8 \
  --skip_existing
```

### 15.5 `--sample_prompts` 报错

正常。当前 NanoSaur 最小实现暂不支持训练中 sample。

先不要传 `--sample_prompts`。

### 15.6 Windows 上 text encoder cache 段错误或 Permission denied

本地 Windows smoke test 中发现两个独立问题：

- `sentencepiece==0.2.1` 读取 NanoSaur text encoder safetensors 内置的 `spiece_model` 时可能在原生层段错误；
- 旧代码通过 `NamedTemporaryFile` 落盘再让 SentencePiece 打开，在 Windows 上会因为临时文件仍被当前进程占用而触发 `Permission denied`。Linux 通常允许这种文件访问模式，但直接从内存加载 serialized proto 更不依赖操作系统文件语义。

处理方式：

1. 在项目 venv 内使用 `sentencepiece==0.1.99`，规避本地观察到的 `0.2.1` native segfault；
2. 当前代码已改为 `SentencePieceProcessor.LoadFromSerializedProto(...)`，不再通过临时文件加载 tokenizer，Windows/Linux 都可用。

性能影响：tokenizer 只在 text encoder 初始化时加载一次；`LoadFromSerializedProto` 避免临时文件 I/O，且当前实现直接从 `uint8` tensor 转 bytes，不会影响训练迭代性能。

### 15.7 text cache 全是 NaN

本地 RTX 5080 测试中，`--text_encoder_dtype fp16` 会让 Gemma text encoder 输出 NaN，保存逻辑会把 NaN 替换为 0，导致文本条件基本失效。

请使用：

```bash
--text_encoder_dtype bf16
```

或使用 `fp32`。训练本身仍建议 `--mixed_precision bf16`。

### 15.8 保存 LoRA 时报 `Unknown architecture: ns`

这是 metadata 注册缺失导致的代码问题。当前已在 `sai_model_spec.py` 中注册 NanoSaur metadata：

```text
modelspec.architecture = Nanosaur-1.2B/lora
ss_base_model_version = nanosaur
```

### 15.9 backward 报 `Found dtype Half but expected Float`

这是 NanoSaur trainer 中 `model_pred` 与 `target` dtype 未统一导致的反传错误。当前已在 `nanosaur_train_network.py` 中将二者统一到 `network_dtype` 后再计算 loss。

### 15.10 分桶步长为什么是 16？要不要改成 64？

NanoSaur 的默认 bucket 步长是 16，来源于原生 VAE / DINO patch16 的空间对齐要求：输入宽高至少需要按 16 对齐，VAE latent 空间约为 `H/16 x W/16`。因此 16 是最小硬约束。

不建议把默认值硬改成 64。64 是 16 的倍数，模型层面通常安全，但它只是更粗的分桶策略，会减少 bucket 数、改变 resize/crop 粒度，对极端宽高比图片可能降低分辨率利用率。如果确实需要 64，应作为额外可配置项实现，并校验它是 16 的倍数。

当前示例使用：

```toml
enable_bucket = true
bucket_no_upscale = false
```

`bucket_no_upscale = false` 是默认值，显式写出是为了避免误解。

### 15.11 保存的 LoRA 是 bf16 吗？为什么文件看起来变大？

当前 NanoSaur trainer 保存 LoRA 时使用 `dit_dtype` 作为保存 dtype。通常命令里传：

```bash
--mixed_precision bf16
```

且不额外覆盖 `--dit_dtype` 时，`dit_dtype` 会被设置为 bf16，`lora_nanosaur.py` 会在保存前把浮点 LoRA tensor cast 到 bf16。因此单个 LoRA 文件正常应是 bf16，不是 fp32。

如果觉得文件或输出目录变大，通常是这些原因：

1. `--network_dim` 变大。LoRA 文件大小基本跟 rank 线性相关，rank 8 约为 rank 4 的 2 倍，rank 16 约为 rank 4 的 4 倍；
2. 每次保存会同时生成训练格式和 ComfyUI 格式两份：`*.safetensors` 与 `*-comfyui.safetensors`；
3. `--save_every_n_epochs 1` 会每个 epoch 保存一次，训练结束还会保存最终版本，所以整个输出目录会累积多组文件；
4. `--network_alpha` 不明显影响文件大小，它主要影响 LoRA scale。

可以用下面命令检查某个 LoRA 文件里的 dtype：

```bash
python -c "from safetensors.torch import load_file; p='path/to/lora.safetensors'; sd=load_file(p, device='cpu'); print(sorted({str(v.dtype) for v in sd.values()})); print(len(sd))"
```

如果输出包含 `torch.bfloat16`，说明保存的是 bf16。ComfyUI 文件会多出 alpha tensor，tensor 数量比训练格式更多，这是正常的。

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
  --dit ../model/nanosaur_diffusion_model.safetensors \
  --vae ../model/nanosaur_vae_decoder.safetensors \
  --uncond_text_embedding ../cache/nanosaur_1024_fp16/uncond_ns_te.safetensors \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 8 \
  --network_alpha 4 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 100 \
  --save_every_n_steps 50 \
  --output_dir ../outputs/nanosaur_smoke_test \
  --output_name nanosaur_smoke
```

如果成功，输出目录中应出现：

```text
nanosaur_smoke-000000050.safetensors
nanosaur_smoke-000000050-comfyui.safetensors
```

### 16.1 本地验证记录

在 `<repo-root>` 的项目 venv 中完成过一次真实 smoke test：

- GPU：RTX 5080 16GB；
- Python：项目内 `venv`；
- PyTorch：`2.8.0+cu128`；
- `sentencepiece`：`0.1.99`；
- dataset：97 张图片，其中 1 张缺同名 `.txt`，当前数据加载仍会为其生成空 caption cache；
- latent cache：97 个 `*_ns.safetensors`；
- text cache：97 个 `*_ns_te.safetensors` + `uncond_ns_te.safetensors`；
- 训练：`max_train_steps=1`，`network_dim=4`，`mixed_precision=bf16`，`optimizer_type=AdamW`；
- 额外验证：`optimizer_type=Adafactor` 可完成 1 step 训练并保存 LoRA；optimizer/scheduler 仍沿用 musubi 通用逻辑，但 `bitsandbytes`、`wandb`、`tensorboard` 等可选能力取决于环境是否安装对应依赖；
- 输出目录：`outputs/nanosaur_smoke_lora/`。

已验证生成并可加载：

```text
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-comfyui.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-step00000001.safetensors
outputs/nanosaur_smoke_lora/nanosaur_smoke_lora-step00000001-comfyui.safetensors
```

其中训练格式 LoRA 约 7.4 MB，包含 266 个 LoRA tensor；ComfyUI 导出约 7.4 MB，包含 399 个 tensor，metadata 中 `modelspec.architecture` 为 `Nanosaur-1.2B/lora`。

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
