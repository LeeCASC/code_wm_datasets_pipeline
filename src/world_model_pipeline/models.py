"""Portable model locations and integrity checks; weights remain outside Git."""
import hashlib
from pathlib import Path

from .io import read_json
from .paths import ROOT, project_path

MODEL_FILES = {
    'dit': 'diffusion_models/minimax_h3_ref2va_bf16.safetensors',
    'text_encoder': 'text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors',
    'video_vae': 'vae/minimax_h3_video_vae_fp16.safetensors',
    'audio_vae': 'vae/minimax_h3_audio_vae_fp32.safetensors',
    'processor': 'processors/Qwen3-VL-32B-Instruct',
    'lora': 'lora/minimax-h3-cwm.safetensors',
}


def model_paths(model_root=None, *, cwm=True):
    folder = project_path(model_root) if model_root else ROOT / 'models'
    return {key: str(folder / relative) for key, relative in MODEL_FILES.items() if cwm or key != 'lora'}


def verify_models(model_root=None, *, full=False):
    folder = project_path(model_root) if model_root else ROOT / 'models'
    manifest = read_json(ROOT / 'configs/model_manifest.json')
    if manifest is None:
        raise FileNotFoundError('configs/model_manifest.json is missing; finish copying the model assets first')
    results = []
    for row in manifest['files']:
        path = folder / row['path']
        valid = path.is_file() and path.stat().st_size == row['bytes']
        actual = None
        if valid and full:
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                    hasher.update(block)
            actual = hasher.hexdigest()
            valid = actual == row['sha256']
        results.append({'path': row['path'], 'valid': valid, 'sha256': actual})
    return {'status': 'pass' if all(row['valid'] for row in results) else 'failed',
            'model_root': str(folder), 'full_sha256_checked': full, 'files': results}
