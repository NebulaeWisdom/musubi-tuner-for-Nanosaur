#!/usr/bin/env bash
set -euo pipefail

DATASET_CONFIG=${DATASET_CONFIG:-examples/nanosaur_dataset.toml}
MODEL_DIR=${MODEL_DIR:-../model}
CACHE_DIR=${CACHE_DIR:-../cache/nanosaur_1024_fp16}
OUTPUT_DIR=${OUTPUT_DIR:-../outputs/nanosaur_musubi_lora}

python -m musubi_tuner.nanosaur_cache_latents \
  --dataset_config "${DATASET_CONFIG}" \
  --vae "${MODEL_DIR}/nanosaur_vae_decoder.safetensors" \
  --vae_dtype fp16 \
  --batch_size 4 \
  --skip_existing

python -m musubi_tuner.nanosaur_cache_text_encoder_outputs \
  --dataset_config "${DATASET_CONFIG}" \
  --text_encoder "${MODEL_DIR}/nanosaur_text_encoder.safetensors" \
  --text_encoder_dtype bf16 \
  --batch_size 8 \
  --skip_existing

accelerate launch --num_processes 1 \
  -m musubi_tuner.nanosaur_train_network \
  --dataset_config "${DATASET_CONFIG}" \
  --dit "${MODEL_DIR}/nanosaur_diffusion_model.safetensors" \
  --vae "${MODEL_DIR}/nanosaur_vae_decoder.safetensors" \
  --uncond_text_embedding "${CACHE_DIR}/uncond_ns_te.safetensors" \
  --network_module musubi_tuner.networks.lora_nanosaur \
  --network_dim 16 \
  --network_alpha 4 \
  --network_dropout 0 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --sdpa \
  --max_train_steps 1000 \
  --save_every_n_steps 500 \
  --output_dir "${OUTPUT_DIR}" \
  --output_name nanosaur_lora
