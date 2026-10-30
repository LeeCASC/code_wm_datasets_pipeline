# Third-Party Notices

This file records the principal external components required by CWM MiniMax H3
Inference. It is not a substitute for reviewing the complete license set of an
installed Python/CUDA environment.

## FlashAttention-3

- Project: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- Required revision: `c68c592fd9da1e40a4fb0b56229caae6754ac5c9`
- Package built by this guide: `hopper/` (`flash-attn-3`)
- License: BSD 3-Clause License

The pinned Hopper build also downloads fixed NVIDIA compiler components as
described in its upstream `hopper/setup.py`. Those components and CUDA retain
their own license terms and are not covered by this repository's Apache
License 2.0.

## Musubi Tuner

- Project: [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
- Required revision: `c4dc6b8038c114f730f1aa24a98064de3bd62437`
- Integration patches: `patches/minimax-h3-retake34.patch` and
  `patches/musubi-headless-opencv.patch`
- License: Apache License 2.0 for the relevant MiniMax H3 code, subject to the
  copyright and provenance notices retained in the upstream source files.

The patch modifies a separately obtained Musubi checkout. It does not change
the license of Musubi or any upstream-derived code. Musubi's MiniMax H3
implementation also documents source adapted from Hugging Face Diffusers; the
upstream copyright and Apache-2.0 headers remain authoritative.

## MiniMax H3 model artifacts

- Model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Validated single-file distribution:
  [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/0543966fbdce5ba05709a8f2031c94bdba629b4a)
- Validated distribution revision:
  `0543966fbdce5ba05709a8f2031c94bdba629b4a`
- License: [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

MiniMax H3 transformer, VAE, and related model weights are not included in
this repository and are not licensed under this repository's Apache License
2.0. Users must obtain them separately and comply with the MiniMax-H3
Community License.

The processor/tokenizer files used by the validated setup come from
[Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb)
at revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb` and retain their own
Apache-2.0 license.

## CWM LoRA checkpoint

The released CWM LoRA checkpoint is distributed separately from this source
repository through the
[AWM MiniMax H3 LoRA checkpoint repository](https://huggingface.co/NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints).
The installation script selects the released checkpoint by its SHA-256 and
installs it locally as `models/lora/minimax-h3-cwm.safetensors`. The checkpoint
is provided under
the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
and is not covered by this source repository's Apache License 2.0.

## CWM inference examples

The released RGB first frames, prompts, and compact Depth + Semantic-ID inputs
are distributed separately through
[code-world-model-inference-examples-40](https://huggingface.co/datasets/NTU-yiwen/code-world-model-inference-examples-40).
The installation script expands these inputs locally; they are not included in
this source repository and are not covered by its Apache License 2.0.

## Python and CUDA dependencies

The runtime imports Musubi Tuner, PyTorch, PyAV, NumPy, Pillow, Safetensors,
and their transitive dependencies. These packages are installed separately
and retain their own copyright notices and license terms. No dependency's
license is replaced by this repository's Apache License 2.0.
