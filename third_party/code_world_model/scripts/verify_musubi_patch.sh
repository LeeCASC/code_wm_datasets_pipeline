#!/usr/bin/env bash
set -euo pipefail

readonly BASE_COMMIT="c4dc6b8038c114f730f1aa24a98064de3bd62437"
readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PATCH_PATH="${SCRIPT_DIR}/../patches/minimax-h3-retake34.patch"
readonly HEADLESS_PATCH_PATH="${SCRIPT_DIR}/../patches/musubi-headless-opencv.patch"
readonly -a EXPECTED_PATHS=(
  "pyproject.toml"
  "src/musubi_tuner/minimax_h3/model.py"
  "src/musubi_tuner/minimax_h3/packing.py"
  "src/musubi_tuner/minimax_h3/sampling.py"
)

usage() {
  printf 'Usage: %s /path/to/musubi-tuner\n' "${0##*/}" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

if [[ ! -f "${PATCH_PATH}" ]]; then
  printf 'Patch not found: %s\n' "${PATCH_PATH}" >&2
  exit 1
fi
if [[ ! -f "${HEADLESS_PATCH_PATH}" ]]; then
  printf 'Patch not found: %s\n' "${HEADLESS_PATCH_PATH}" >&2
  exit 1
fi

MUSUBI_REPO="$(git -C "$1" rev-parse --show-toplevel)"
readonly MUSUBI_REPO
RESOLVED_COMMIT="$(git -C "${MUSUBI_REPO}" rev-parse --verify "${BASE_COMMIT}^{commit}")"
readonly RESOLVED_COMMIT
if [[ "${RESOLVED_COMMIT}" != "${BASE_COMMIT}" ]]; then
  printf 'Resolved baseline differs from expected commit: %s\n' "${RESOLVED_COMMIT}" >&2
  exit 1
fi

DEFAULT_TMP_ROOT="${SCRIPT_DIR}/../tmp"
mkdir -p "${TMPDIR:-${DEFAULT_TMP_ROOT}}"
CHECK_ROOT="$(mktemp -d "${TMPDIR:-${DEFAULT_TMP_ROOT}}/cwm-h3-patch-check.XXXXXX")"
WORKTREE="${CHECK_ROOT}/musubi"

cleanup() {
  if [[ -e "${WORKTREE}/.git" ]]; then
    git -C "${MUSUBI_REPO}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  fi
  rmdir "${CHECK_ROOT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git -C "${MUSUBI_REPO}" worktree add --detach "${WORKTREE}" "${BASE_COMMIT}" >/dev/null
git -C "${WORKTREE}" apply --check "${PATCH_PATH}"
git -C "${WORKTREE}" apply "${PATCH_PATH}"
git -C "${WORKTREE}" apply --check "${HEADLESS_PATCH_PATH}"
git -C "${WORKTREE}" apply "${HEADLESS_PATCH_PATH}"
git -C "${WORKTREE}" diff --check

mapfile -t changed_paths < <(git -C "${WORKTREE}" diff --name-only)
if [[ "${changed_paths[*]}" != "${EXPECTED_PATHS[*]}" ]]; then
  printf 'Unexpected patched paths:\n' >&2
  printf '  %s\n' "${changed_paths[@]}" >&2
  exit 1
fi

printf 'Verified clean apply at %s; patched paths: %s\n' \
  "${BASE_COMMIT}" "${changed_paths[*]}"
