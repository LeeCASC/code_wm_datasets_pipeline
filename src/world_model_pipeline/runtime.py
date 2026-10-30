"""Local/offline model execution and explicit subprocess module entry points."""
import os
from pathlib import Path
import sys

from .paths import ROOT

PROXY_KEYS = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


def clean_environment(gpu=None, offline=True):
    environment = dict(os.environ)
    for key in (*PROXY_KEYS, 'TRANSFORMERS_CACHE'):
        environment.pop(key, None)
    cache = ROOT / '.cache/huggingface'
    environment.update(
        CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT / 'src'),
        HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'),
        HF_HUB_DISABLE_TELEMETRY='1', WORLD_MODEL_PIPELINE_ROOT=str(ROOT),
    )
    if offline:
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return environment


def configure_local_runtime():
    environment = clean_environment(offline=True)
    # Keep the caller's explicit CUDA visibility; this function does not select a GPU.
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    for key in (*PROXY_KEYS, 'TRANSFORMERS_CACHE'):
        os.environ.pop(key, None)
    os.environ.update(environment)
    (ROOT / '.cache/huggingface').mkdir(parents=True, exist_ok=True)


def module_command(module, *arguments):
    return [sys.executable, '-m', 'world_model_pipeline.' + module, *map(str, arguments)]
