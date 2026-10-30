# MiniMax H3 Inference

This repository contains the MiniMax H3 Ref2VA inference component and tuned
CWM LoRA weights used in Code World Model. It generates video from an RGB
first-frame anchor, per-frame Depth + Semantic-ID conditions, and per-window
text prompts, with Retake34 continuation across multiple windows.

Current coding agents can effectively adapt and extend existing game code, but
still struggle to implement reliable collision handling, driving systems, and
related simulation logic from scratch. We therefore built our coding-agent
pipeline on an existing closed-source game-code foundation. Because this
foundation cannot be redistributed, we are unable to release the tools used to
generate the proxy inputs (i.e., Depth and Semantic-ID conditions), and we
apologize for this limitation.

We instead provide a large collection of multi-window proxy examples for direct
experimentation. To explore a different visual style or content, use an
image-generation or editing agent to modify the style and content of a provided
RGB first frame. Then use a coding agent to rewrite the corresponding original
prompts so that they align with the edited first frame. Preserve each prompt's
original structure and change only the necessary style and content, as changing
the structure generally produces worse results.

## Environment setup

This project was tested on one NVIDIA H800 80 GB GPU. The setup below keeps
the versions used in our experiments, but does not require a particular Conda,
venv, or cache layout.

## Tested environment

| Component | Tested version |
| --- | --- |
| OS | Linux |
| GPU | NVIDIA H800 80 GB (SM90) |
| Python | 3.10.15 |
| CUDA toolkit used to build FA3 | 12.4 |
| PyTorch CUDA runtime | 12.8 |
| FlashAttention-3 source | `c68c592fd9da1e40a4fb0b56229caae6754ac5c9` |
| Musubi Tuner source | `c4dc6b8038c114f730f1aa24a98064de3bd62437` |

The main Python package versions were:

| Package | Version | Package | Version |
| --- | ---: | --- | ---: |
| `torch` | `2.9.1+cu128` | `torchvision` | `0.24.1+cu128` |
| `triton` | `3.5.1` | `transformers` | `4.57.6` |
| `diffusers` | `0.32.1` | `accelerate` | `1.6.0` |
| `bitsandbytes` | `0.49.1` | `huggingface-hub` | `0.34.3` |
| `av` | `14.0.1` | `opencv-python-headless` | `4.10.0.84` |
| `numpy` | `2.2.6` | `Pillow` | `12.1.0` |
| `safetensors` | `0.4.5` | `sentencepiece` | `0.2.1` |
| `tokenizers` | `0.22.2` | `einops` | `0.7.0` |
| `ftfy` | `6.3.1` | `easydict` | `1.13` |
| `toml` | `0.10.2` | `tqdm` | `4.67.1` |
| `voluptuous` | `0.15.2` | `psutil` | `7.2.1` |

The complete direct dependency list, including build tools, is pinned in
[`requirements-h800-cu128-py310.txt`](requirements-h800-cu128-py310.txt).

## Install

The machine needs Git, a C/C++ compiler, Python 3.10 development headers, and
the CUDA 12.4 toolkit with `nvcc`.

From the repository root, create any Python 3.10 environment you prefer and
install the pinned packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-h800-cu128-py310.txt
```

Install the exact FlashAttention-3 revision used in the experiments:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git ../flash-attention
git -C ../flash-attention checkout --detach \
  c68c592fd9da1e40a4fb0b56229caae6754ac5c9
git -C ../flash-attention submodule update --init csrc/cutlass

CUDA_HOME=/usr/local/cuda-12.4 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
NVCC_THREADS=2 MAX_JOBS=8 \
python -m pip install --no-build-isolation --no-deps \
  ../flash-attention/hopper
```

Install the exact Musubi revision with this repository's two patches, then
install this project:

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git ../musubi-tuner
git -C ../musubi-tuner checkout --detach \
  c4dc6b8038c114f730f1aa24a98064de3bd62437

./scripts/verify_musubi_patch.sh ../musubi-tuner
git -C ../musubi-tuner apply "$(pwd)/patches/minimax-h3-retake34.patch"
git -C ../musubi-tuner apply "$(pwd)/patches/musubi-headless-opencv.patch"

python -m pip install --no-build-isolation --no-deps ../musubi-tuner
python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

An optional GPU smoke test is available after installation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/verify_h800_runtime.py
```

## Model, LoRA, and inference examples

This source repository keeps large release files external. Model and CWM LoRA
weights are available from the sources below. Run the installation script from
the source-repository root:

```bash
./scripts/install_release_assets.sh "$PWD/release"
```

The script uses only the already published release assets. It downloads the 40
compact multi-window inputs from
[NTU-yiwen/code-world-model-inference-examples-40](https://huggingface.co/datasets/NTU-yiwen/code-world-model-inference-examples-40)
at revision `f8307bf50cb19bed191682734fdf444738160f91`. It downloads the released
CWM LoRA from the public
[NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints](https://huggingface.co/NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints)
repository at revision `a3df90f6742748896457cec84b13dbedbce56fff`, verifies its SHA-256,
and installs it as `models/lora/minimax-h3-cwm.safetensors`.

The compact dataset stores conditions in lossless NPZ chunks. The installation
script calls `scripts/materialize_hf_examples.py` to expand them locally into
the exact `.depth.f32` and `.semantic_id.png` layout consumed by inference. For
the published uint16 fixed-log Depth inputs, the conversion chooses float32
depth values that round-trip to every original uint16 code exactly. The script
also creates each runnable configuration from the published prompts and the
fixed release seeds included with this source repository, then verifies all
prompts, window starts, seeds, and paths before accepting an example. No private
Hugging Face repository is required.

After the script finishes, the release directory contains:

```text
release/
  downloads/code-world-model-inference-examples-40/
  examples/example_XX/config.json
  examples/example_XX/first_frame.png
  examples/example_XX/conditions/*.depth.f32
  examples/example_XX/conditions/*.semantic_id.png
  models/lora/minimax-h3-cwm.safetensors
  models/diffusion_models/...
  models/text_encoders/...
  models/vae/...
  models/processors/Qwen3-VL-32B-Instruct/...
```

Internally, the script downloads the public base-model files from the same fixed
revisions used in our experiments:

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --revision 0543966fbdce5ba05709a8f2031c94bdba629b4a \
  --local-dir models

hf download Qwen/Qwen3-VL-32B-Instruct \
  chat_template.json config.json generation_config.json merges.txt \
  preprocessor_config.json tokenizer_config.json tokenizer.json \
  video_preprocessor_config.json vocab.json \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
  --local-dir models/processors/Qwen3-VL-32B-Instruct
```

Each prepared example contains the released RGB first frame, per-frame Depth +
Semantic-ID proxy conditions, original per-window prompts, seed, and runnable
configuration. The tested CWM LoRA loads 200 adapter modules. Reproducing a
specific result requires these unchanged inputs and a newly prepared cache.

The compact Hugging Face dataset itself does not include model weights, caches,
or output videos. The LoRA and public base-model files are downloaded separately
by the installation script; `prepare` creates caches locally.

The Audio VAE is required even though the final MP4 is silent. For input
formats, configuration examples, cache rules, and commands, continue to the
[inference guide](INFERENCE.md).
