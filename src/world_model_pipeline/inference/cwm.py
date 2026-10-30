#!/usr/bin/env python3
"""Experimental CWM entry point using Musubi's existing torch/SDPA backend.

The upstream package and flash3-only JSON contract stay unchanged. This runner
applies an explicit runtime override and uses separate cache/output paths.
Small kernel checks do not establish full-model parity or memory requirements.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time
from world_model_pipeline.runtime import configure_local_runtime


def runtime_config(source, *, check_files=True):
    from cwm_h3_inference.config import load_config
    base = load_config(source, check_files=check_files)
    return replace(base, attention_mode='torch',
                   cache_dir=base.cache_dir.with_name(base.cache_dir.name+'_a100_sdpa'),
                   output=base.output.with_name(base.output.stem+'.a100_sdpa'+base.output.suffix))


@contextmanager
def sampling_progress(path):
    """Observe Musubi's existing step callback without changing model inputs."""
    sampling = import_module('musubi_tuner.minimax_h3.sampling')
    original = sampling.sample_joint_av
    calls = 0

    @wraps(original)
    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        window_index = calls - 1
        started = time.monotonic()
        previous = kwargs.get('step_callback')

        def callback(step, total):
            if previous is not None:
                previous(step, total)
            record = {'window_index': window_index, 'step': step, 'total_steps': total,
                      'sampler_elapsed_seconds': round(time.monotonic()-started, 2)}
            temporary = path.with_suffix(path.suffix+'.tmp')
            temporary.write_text(json.dumps(record)+'\n')
            temporary.replace(path)
            print(json.dumps({'sampling_progress': record}), flush=True)

        return original(*args, **dict(kwargs, step_callback=callback))

    sampling.sample_joint_av = observed
    try:
        yield
    finally:
        sampling.sample_joint_av = original


def main():
    configure_local_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate','prepare','generate'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--crf', type=int, default=18)
    args = parser.parse_args()
    config = runtime_config(args.config)
    receipt = {'experimental': True, 'upstream_json_attention_mode': 'flash3',
               'runtime_attention_mode': config.attention_mode, 'backend': 'PyTorch SDPA',
               'source_config': str(config.source), 'cache_dir': str(config.cache_dir),
               'output': str(config.output), 'output_frames': config.output_frames,
               'full_model_parity_established': False}
    if args.command == 'validate':
        print(json.dumps(receipt,indent=2))
        return 0
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES')
    receipt['gpu'] = torch.cuda.get_device_name(0)
    receipt['command'] = args.command
    if args.command == 'generate' and config.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing video: {config.output}')
    config.output.parent.mkdir(parents=True, exist_ok=True)
    path = config.output.with_suffix('.runtime.json')
    receipt['status'] = 'started'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        if args.command == 'prepare':
            from cwm_h3_inference.cache import prepare_caches
            result = prepare_caches(config)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from cwm_h3_inference.engine import generate
            from world_model_pipeline.inference.export import quality_export
            # Refuse an implicit quadratic math-kernel fallback for the large H3 sequence.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), sampling_progress(config.output.with_suffix('.progress.json')), quality_export(args.crf):
                result = generate(config)
        receipt['status'] = 'complete'
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'runtime': receipt, 'result': result},indent=2))
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc))
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
