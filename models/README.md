# Local model assets

This folder contains independent copies of the five safetensors weights and the Qwen processor used in the experiments. Large weights and processor contents are Git-ignored. The portable file inventory is `../configs/model_manifest.json`.

```text
diffusion_models/minimax_h3_ref2va_bf16.safetensors
text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
lora/minimax-h3-cwm.safetensors
processors/Qwen3-VL-32B-Instruct/
```

Check sizes with `python scripts/wmp models verify`; read and verify every SHA-256 with `--full`. Copy another existing local model tree with `python scripts/copy_models.py --source /path/to/models`. This tool refuses to overwrite destination files and verifies each independent copy before promotion.
