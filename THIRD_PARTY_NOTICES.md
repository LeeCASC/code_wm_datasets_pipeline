# Third-party sources and model assets

## Vendored source

`third_party/manifest.json` records repository revisions. These directories contain copied tracked source files, not nested Git repositories or submodules.

- `third_party/code_world_model`: https://github.com/buaacyw/code-world-model, revision `17fa82ba2fb8168aa0d30fc51a1d1d0ebbc9180b`. Its Apache-2.0 `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and original documentation are preserved.
- `third_party/musubi_tuner`: https://github.com/kohya-ss/musubi-tuner, revision `c4dc6b8038c114f730f1aa24a98064de3bd62437`, including the already-applied CWM Retake34 and headless OpenCV changes. `third_party/musubi_tuner.local.patch` records the exact local differences against that revision. The Musubi README's License section and individual source notices are preserved, including notices for model-specific subdirectories.

The new package changes entry points and wraps the vendored runtime; it does not silently edit these vendored source snapshots. Python distribution dependencies are listed in `requirements.lock.txt` and are not vendored as an entire virtual environment.

## Model files

Local H3 Ref2VA weights, shared text encoder and VAEs, CWM LoRA, and the Qwen processor are copied under `models/`. They are excluded from Git. `configs/model_manifest.json` records file sizes and hashes; it does not grant a new model license.

- H3 base: MiniMax-H3 / Ref2VA, https://huggingface.co/MiniMaxAI/MiniMax-H3. The local DiT download used the Comfy-Org MiniMax-H3 safetensors distribution.
- Text encoder and VAEs: converted from local MiniMax-H3 FL2VA shared components using the preserved conversion tool and release structure headers. Conversion does not imply bytewise or numerical parity with every upstream package.
- CWM LoRA: https://hf-mirror.com/NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints.
- Processor: Qwen/Qwen3-VL-32B-Instruct files used by the local H3 encoder.

Upstream code and model terms remain attached to their respective artifacts. No new blanket license has been assigned to the project's own code by this refactor.

## Example images

The included appearance anchors were created in the previous experiments using the built-in image generation tool. Receipts describe them as approximate appearance references, not calibrated geometry. The service's underlying image-model weights/version are not available in this repository.
