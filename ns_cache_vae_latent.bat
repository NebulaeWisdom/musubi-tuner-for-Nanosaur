set PYTHONUTF8=1
..\venv\Scripts\python.exe -m musubi_tuner.nanosaur_cache_latents --dataset_config examples\nanosaur_dataset.toml --vae ..\model\nanosaur_vae_decoder.safetensors --vae_dtype fp16 --batch_size 1 --num_workers 1 --skip_existing

pause
