#!/usr/bin/env python3
"""Run unmerged H3 Ref2VA using native Musubi generation and RGB references."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
from PIL import Image

from world_model_pipeline.paths import ROOT
from world_model_pipeline.runtime import configure_local_runtime


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def adapt_video_decode_dtype(vae):
    """Native sampler accumulates FP32; this workstation's VAE is FP16."""
    original = vae.decode
    @functools.wraps(original)
    def decode(latents):
        weight = next(vae.parameters())
        return original(latents.to(device=weight.device, dtype=weight.dtype))
    vae.decode = decode
    return vae


def verify(path):
    timestamps = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if float(stream.average_rate) != 24 or (stream.width,stream.height) != (1344,768):
            raise ValueError('Unexpected generated video geometry')
        for index, frame in enumerate(container.decode(video=0)):
            timestamps.append(float(frame.time))
            if index in [0,62,123]:
                frame.to_image().save(path.parent/f'generated_{index:06d}.png')
    if len(timestamps) != 124 or not np.allclose(np.diff(timestamps), 1/24, atol=1e-6):
        raise ValueError('Generated video has an incorrect frame count or timestamp sequence')
    return {'decoded_frames': 124, 'fps': 24, 'size_wh': [1344,768], 'duration_s': 124/24}


def run(config_path):
    configure_local_runtime()
    import torch
    import imageio_ffmpeg
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from musubi_tuner import minimax_h3_generate_video as generator
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    native, output = Path(config['native_joint_output']), Path(config['output'])
    state = {'id': config['id'], 'task': 'ref2va', 'cwm_lora_applied': False,
             'camera_pose_verdict': 'not_certified', 'strict_neural_pixel_gt': False,
             'rgb_reference_size_wh': [672,384], 'backend': 'PyTorch FLASH SDPA',
             'video_vae_decode_cast': 'FP32 sample converted to FP16 at decoder boundary',
             'upstream_numeric_parity_established': False}
    started = time.monotonic()
    original_functions = {}

    def status(phase, **values):
        state.update(phase=phase, elapsed_seconds=round(time.monotonic()-started), **values)
        save(directory/'status.json', state)
        print(json.dumps({'phase': phase, **values}), flush=True)

    try:
        if output.exists():
            existing = json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
            if existing.get('phase') == 'complete':
                return
            raise FileExistsError(output)
        if not native.exists():
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Expose exactly one CUDA GPU for this job')
            fields = config['models']
            cli = ['--task','ref2va','--dit',fields['dit'],'--video_vae',fields['video_vae'],
                   '--audio_vae',fields['audio_vae'],'--text_encoder',fields['text_encoder'],
                   '--processor',fields['processor'],'--reference_jsonl',config['references_jsonl'],
                   '--width','1344','--height','768','--frame_count','124','--steps',str(config['steps']),
                   '--seed',str(config['seed']),'--blocks_to_swap',str(config.get('blocks_to_swap', 48)),
                   '--use_pinned_memory_for_block_swap','--attn_mode','torch','--output',str(native)]
            args = generator.setup_parser().parse_args(cli)
            if args.lora_weight:
                raise RuntimeError('The raw RGB reference baseline must not load the CWM LoRA')

            def observe(name, phase):
                original = getattr(generator, name)
                original_functions[name] = original
                @functools.wraps(original)
                def wrapped(*a, **kw):
                    status(phase)
                    return original(*a, **kw)
                setattr(generator, name, wrapped)

            observe('_encode_text', 'encoding_text')
            observe('_encode_visual_conditions', 'encoding_rgb_references')
            observe('load_h3_transformer', 'loading_transformer')
            original_load_vae = generator.load_video_vae
            original_functions['load_video_vae'] = original_load_vae
            def load_video_vae(*a, **kw):
                return adapt_video_decode_dtype(original_load_vae(*a, **kw))
            generator.load_video_vae = load_video_vae
            original_sample = generator.sample_joint_av
            original_functions['sample_joint_av'] = original_sample
            identity = hashlib.sha256(config_path.read_bytes())
            for artifact in [Path(config['anchor']), Path(config['source_reference']), Path(config['references_jsonl'])]:
                identity.update(artifact.read_bytes())
            identity = identity.hexdigest()
            recovery = directory/'sample_recovery.safetensors'

            @functools.wraps(original_sample)
            def sample(*a, **kw):
                from safetensors import safe_open
                from safetensors.torch import load_file, save_file
                from musubi_tuner.minimax_h3.sampling import H3SampleResult
                if recovery.exists():
                    with safe_open(str(recovery),framework='pt',device='cpu') as handle:
                        if (handle.metadata() or {}).get('experiment_sha256') != identity:
                            raise ValueError('Sample recovery belongs to different inputs')
                    values = load_file(str(recovery),device='cpu')
                    if not all(torch.isfinite(value).all().item() for value in values.values()):
                        raise ValueError('Saved sample contains non-finite values')
                    status('decoding_saved_sample')
                    return H3SampleResult(video=values['video'],audio=values['audio'])
                status('sampling', packed_rows=kw['layout'].row_count)
                old_callback = kw.get('step_callback')
                sampling_start = time.monotonic()
                def callback(step, total):
                    if old_callback:
                        old_callback(step,total)
                    progress = {'step': step, 'total_steps': total,
                                'sampling_elapsed_seconds': round(time.monotonic()-sampling_start,2)}
                    save(directory/'sampling_progress.json', progress)
                    status('sampling', **progress)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = original_sample(*a, **dict(kw,step_callback=callback))
                if not torch.isfinite(result.video).all().item() or not torch.isfinite(result.audio).all().item():
                    raise ValueError('H3 sampling produced non-finite latents')
                temporary = recovery.with_suffix('.safetensors.tmp')
                save_file({'video':result.video.detach().cpu().contiguous(),
                           'audio':result.audio.detach().cpu().contiguous()}, str(temporary),
                          metadata={'experiment_sha256':identity,'task':'native_rgb_ref2va'})
                temporary.replace(recovery)
                status('decoding_audio_video')
                return result
            generator.sample_joint_av = sample
            status('validating')
            generator.run_generation(args)
        status('validating_native_video')
        verify(native)
        status('exporting_silent_video')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner','-loglevel','error','-nostdin',
                        '-n','-i',str(native),'-map','0:v:0','-c:v','copy','-an',str(output)], check=True)
        result = verify(output)
        with av.open(str(output)) as container:
            if container.streams.audio:
                raise ValueError('The RGB deliverable unexpectedly contains audio')
        result.update(output=str(output), task='ref2va', cwm_lora_applied=False,
                      camera_pose_verdict='not_certified', strict_neural_pixel_gt=False,
                      source_camera_mapping=str(directory/'source_camera_mapping.json'),
                      placeholder_audio_used_as_condition=False)
        save(directory/'video_validation.json', result)
        status('complete', result=result)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        for name, function in original_functions.items():
            setattr(generator, name, function)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config.resolve())
