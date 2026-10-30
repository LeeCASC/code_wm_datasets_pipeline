#!/usr/bin/env python3
"""Durable GPU queue for the eight prompts and two cuboid controls."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw

from world_model_pipeline.paths import ROOT
from world_model_pipeline.runtime import clean_environment
from world_model_pipeline.runtime import module_command
from world_model_pipeline.orchestration.gpu import gpu_snapshot, eligible_gpus


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def grid(jobs, output, columns):
    inputs = [av.open(job['output']) for job in jobs]
    count = 0
    try:
        with av.open(str(output), 'w') as writer:
            stream = writer.add_stream('libx264', rate=24)
            stream.width = 448*columns
            stream.height = 280*math.ceil(len(jobs)/columns)
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '19', 'preset': 'fast', 'threads': '4'}
            decoders = [container.decode(video=0) for container in inputs]
            while True:
                frames = [next(decoder, None) for decoder in decoders]
                if all(frame is None for frame in frames):
                    break
                if any(frame is None for frame in frames):
                    raise ValueError('Grid inputs have different decoded frame counts')
                canvas = Image.new('RGB', (stream.width, stream.height), '#151515')
                draw = ImageDraw.Draw(canvas)
                for index, (frame, job) in enumerate(zip(frames, jobs)):
                    x, y = index % columns * 448, index // columns * 280
                    canvas.paste(frame.to_image().resize((448, 256), Image.Resampling.LANCZOS), (x, y+24))
                    draw.text((x+8, y+6), job['id']+' | '+job['title'], fill='white')
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                    writer.mux(packet)
                if count in [0, 62, 123]:
                    canvas.save(output.with_name(output.stem+f'_{count:06d}.jpg'), quality=94)
                count += 1
            for packet in stream.encode():
                writer.mux(packet)
    finally:
        for container in inputs:
            container.close()
    if count != 124:
        raise ValueError(f'Expected 124 comparison frames, got {count}')
    return count


def export(directory, manifest, states):
    completed = [job for job in manifest['jobs'] if states[job['id']]['phase'] == 'complete']
    summaries = []
    for job in completed:
        run = Path(job['config']).parent
        config = read(Path(job['config']))
        audit_output = run/'geometry_audit'
        subprocess.run(module_command('diagnostics.geometry',
                        '--video', job['output'], '--camera', str(run/'camera_conditioning.json'),
                        '--conditions', config['condition_root'], '--output', str(audit_output)),
                       env=clean_environment(offline=True), cwd=ROOT, check=True)
        summaries.append({'id': job['id'], 'geometry_diagnostic': str(audit_output/'geometry_diagnostic.json'),
                          'summary': read(audit_output/'geometry_diagnostic.json')['summary']})
    products = {}
    prompt_jobs = [job for job in manifest['jobs'][:8] if states[job['id']]['phase'] == 'complete']
    if len(prompt_jobs) == 8:
        products['prompt_grid'] = str(directory/'prompt_grid.mp4')
        grid(prompt_jobs, Path(products['prompt_grid']), 4)
    box_ids = ['01_terminal_day', '09_box_photo', '10_box_only']
    controls = [next(job for job in manifest['jobs'] if job['id'] == label) for label in box_ids]
    if all(states[job['id']]['phase'] == 'complete' for job in controls):
        products['box_comparison'] = str(directory/'box_comparison.mp4')
        grid(controls, Path(products['box_comparison']), 3)
    report = {'phase': 'complete' if len(completed) == len(manifest['jobs']) else 'complete_with_failures',
              'completed_videos': len(completed), 'total_videos': len(manifest['jobs']),
              'products': products, 'jobs': states, 'geometry_diagnostics': summaries,
              'camera_pose_verdict': 'not_certified', 'strict_neural_pixel_gt': False,
              'prompt_following': 'Not automatically graded. Inspect whether requested weather and background objects actually appear.',
              'limitations': [manifest['anchor_policy'], manifest['background_policy'],
                              'Static-plane diagnostics can be invalid on new nonplanar background buildings; they do not recover 6DoF camera pose.']}
    save(directory/'suite_report.json', report)
    return report


def run(directory):
    manifest = read(directory/'suite_manifest.json')
    active, states = {}, {}
    jobs = manifest['jobs']
    # Get both geometric controls early, rather than waiting behind eight scenes.
    order = ['01_terminal_day', '09_box_photo', '10_box_only']+[job['id'] for job in jobs[1:8]]
    jobs_by_id = {job['id']: job for job in jobs}
    for job in jobs:
        status = read(Path(job['config']).parent/'status.json', {})
        states[job['id']] = {'phase': 'complete' if status.get('phase') == 'complete' and Path(job['output']).is_file() else 'queued'}
    started = time.monotonic()
    status_path = directory/'batch_status.json'
    try:
        while True:
            for label, (child, gpu) in list(active.items()):
                work = Path(jobs_by_id[label]['config']).parent
                status = read(work/'status.json', {})
                progress = read(work/'aircraft_takeoff.a100_sdpa.progress.json')
                states[label] = {'phase': status.get('phase', 'starting'), 'gpu': gpu,
                                 'worker_pid': child.pid, 'sampling': progress}
                if child.poll() is not None:
                    states[label]['phase'] = 'complete' if child.returncode == 0 and status.get('phase') == 'complete' else 'failed'
                    if states[label]['phase'] == 'failed':
                        states[label]['error'] = status.get('error', f'Worker exited with code {child.returncode}')
                    del active[label]
            terminal = sum(state['phase'] in ['complete', 'failed'] for state in states.values())
            if terminal == len(jobs):
                break
            if (directory/'STOP').exists():
                for label in active:
                    (Path(jobs_by_id[label]['config']).parent/'STOP').touch()
                save(status_path, {'phase': 'stop_requested', 'jobs': states})
                return
            snapshot = gpu_snapshot()
            available = eligible_gpus(snapshot, {gpu for _, gpu in active.values()},
                                      manifest['allowed_gpus'], manifest['minimum_free_memory_mib'])
            for gpu in available:
                label = next((label for label in order if states[label]['phase'] == 'queued'), None)
                if label is None:
                    break
                work = Path(jobs_by_id[label]['config']).parent
                with (work/'supervisor.log').open('a') as log:
                    child = subprocess.Popen(module_command('orchestration.supervisor',
                                              '--config', jobs_by_id[label]['config'], '--gpu', str(gpu)),
                                             cwd=ROOT, env=clean_environment(gpu, offline=True),
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                (work/'supervisor.pid').write_text(str(child.pid)+'\n')
                active[label] = child, gpu
                states[label] = {'phase': 'starting', 'gpu': gpu, 'worker_pid': child.pid}
                print(json.dumps({'started': label, 'gpu': gpu, 'pid': child.pid}), flush=True)
            save(status_path, {'phase': 'running' if active else 'waiting_for_free_gpu',
                               'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'elapsed_seconds': round(time.monotonic()-started), 'jobs': states,
                               'completed': sum(state['phase'] == 'complete' for state in states.values()),
                               'failed': sum(state['phase'] == 'failed' for state in states.values()),
                               'total': len(jobs), 'gpu_snapshot': snapshot,
                               'strict_neural_pixel_gt': False})
            if time.monotonic()-started > 172800:
                raise TimeoutError('GPU queue exceeded 48 hours')
            time.sleep(15)
        save(status_path, {'phase': 'exporting', 'jobs': states})
        report = export(directory, manifest, states)
        save(status_path, {'phase': report['phase'], 'completed': report['completed_videos'],
                           'total': len(jobs), 'report': str(directory/'suite_report.json'), 'jobs': states})
    except Exception as exc:
        save(status_path, {'phase': 'failed', 'error': repr(exc), 'jobs': states})
        raise
    finally:
        for child, _gpu in active.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    directory = parser.parse_args().directory.resolve()
    with (directory/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(directory)
