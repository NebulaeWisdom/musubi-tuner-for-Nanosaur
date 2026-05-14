set PYTHONUTF8=1
..\venv\Scripts\python.exe -m musubi_tuner.nanosaur_cache_text_encoder_outputs --dataset_config examples\nanosaur_dataset.toml --text_encoder ..\model\nanosaur_text_encoder.safetensors --text_encoder_dtype bf16 --batch_size 8 --skip_existing

pause
