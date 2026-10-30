#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ROOT="${1:-${REPO_ROOT}/release}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXAMPLES_REPO="NTU-yiwen/code-world-model-inference-examples-40"
EXAMPLES_REVISION="f8307bf50cb19bed191682734fdf444738160f91"
LORA_REPO="NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints"
LORA_REVISION="a3df90f6742748896457cec84b13dbedbce56fff"
LORA_FILENAME="minimax-h3-new1344-q4-system-chat-fresh-official-3ep-warmup100-cosine2e5to1e6-fa3-v1.safetensors"
LORA_SHA256="0e14ad22ae203f684ba6521e0b6fbf58f64668ed454fbce346acafd7f448fdb2"
COMPACT_ROOT="${RELEASE_ROOT}/downloads/code-world-model-inference-examples-40"
LORA_TARGET="${RELEASE_ROOT}/models/lora/minimax-h3-cwm.safetensors"

command -v hf >/dev/null 2>&1 || {
  echo "Missing Hugging Face CLI: install huggingface-hub and ensure 'hf' is on PATH." >&2
  exit 1
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

mkdir -p "${RELEASE_ROOT}"

sha256_file() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

install_lora() {
  if [[ -f "${LORA_TARGET}" ]]; then
    local existing_sha
    existing_sha="$(sha256_file "${LORA_TARGET}")"
    if [[ "${existing_sha}" == "${LORA_SHA256}" ]]; then
      echo "Using verified CWM LoRA at ${LORA_TARGET}"
      return
    fi
    echo "Existing LoRA has an unexpected SHA-256: ${LORA_TARGET}" >&2
    exit 1
  fi

  local stage
  stage="$(mktemp -d "${RELEASE_ROOT}/.lora-download.XXXXXX")"
  trap 'rm -rf -- "${stage}"' RETURN EXIT

  hf download "${LORA_REPO}" \
    --revision "${LORA_REVISION}" \
    --include "**/${LORA_FILENAME}" \
    --local-dir "${stage}"

  local -a matches
  mapfile -d '' matches < <(find "${stage}" -type f -name "${LORA_FILENAME}" -print0)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected one released CWM LoRA, found ${#matches[@]}." >&2
    exit 1
  fi

  local downloaded_sha
  downloaded_sha="$(sha256_file "${matches[0]}")"
  if [[ "${downloaded_sha}" != "${LORA_SHA256}" ]]; then
    echo "Downloaded CWM LoRA failed SHA-256 verification." >&2
    exit 1
  fi

  mkdir -p "$(dirname -- "${LORA_TARGET}")"
  mv -- "${matches[0]}" "${LORA_TARGET}"
  rm -rf -- "${stage}"
  trap - RETURN EXIT
}

hf download "${EXAMPLES_REPO}" \
  --repo-type dataset \
  --revision "${EXAMPLES_REVISION}" \
  --local-dir "${COMPACT_ROOT}"

install_lora

"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_hf_examples.py" \
  --source "${COMPACT_ROOT}" \
  --output "${RELEASE_ROOT}"

hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --revision 0543966fbdce5ba05709a8f2031c94bdba629b4a \
  --local-dir "${RELEASE_ROOT}/models"

hf download Qwen/Qwen3-VL-32B-Instruct \
  chat_template.json config.json generation_config.json merges.txt \
  preprocessor_config.json tokenizer_config.json tokenizer.json \
  video_preprocessor_config.json vocab.json \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
  --local-dir "${RELEASE_ROOT}/models/processors/Qwen3-VL-32B-Instruct"

echo "Release assets are ready at ${RELEASE_ROOT}"
echo "Run from that directory with CONFIG=examples/example_01/config.json"
