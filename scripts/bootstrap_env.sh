#!/usr/bin/env bash
set -euo pipefail
WMP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WMP_PYTHON="${WMP_PYTHON:-python3}"
WMP_ENV="${WMP_ENV:-${WMP_ROOT}/.venv}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
"${WMP_PYTHON}" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10-3.12"'
"${WMP_PYTHON}" -m venv "${WMP_ENV}"
"${WMP_ENV}/bin/python" -m pip install -r "${WMP_ROOT}/requirements.lock.txt"
"${WMP_ENV}/bin/python" -m pip install --no-build-isolation --no-deps -e "${WMP_ROOT}"
"${WMP_ENV}/bin/python" -m pip check
