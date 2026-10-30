#!/usr/bin/env python3
"""Queue four RGB-reference H3 clips and assemble source/enhanced comparisons."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw

from world_model_pipeline.paths import ROOT
from world_model_pipeline.workflows.observers import encode_frames
from world_model_pipeline.runtime import clean_environment
from world_model_pipeline.runtime import module_command
from world_model_pipeline.orchestration.gpu import eligible_gpus, gpu_snapshot


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def frames(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]


def paired(source, generated, path):
    with av.open(str(path), 'w') as writer:
        stream = writer.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 1344, 416, 'yuv420p'
        stream.options = {'crf':'19','preset':'fast','threads':'4'}
        for index, (a,b) in enumerate(zip(source,generated)):
            canvas = Image.new('RGB',(1344,416),'#161616')
            draw = ImageDraw.Draw(canvas)
            for column, (rgb,label) in enumerate([(a,'Source UE RGB'),(b,'Native H3 Ref2VA')]):
                canvas.paste(Image.fromarray(rgb).resize((672,384)),(column*672,32))
                draw.text((column*672+10,10),label+f' | {index/24:.2f}s',fill='white')
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                writer.mux(packet)
            if index in [0,62,119,120,len(source)-1]:
                canvas.save(path.with_name(path.stem+f'_{index:06d}.jpg'), quality=94)
        for packet in stream.encode():
            writer.mux(packet)


def export(directory, manifest, states):
    products = []
    for observer in sorted({job['observer'] for job in manifest['jobs']}):
        jobs = sorted([job for job in manifest['jobs'] if job['observer'] == observer],key=lambda job:job['id'])
        if not all(states[job['id']]['phase'] == 'complete' for job in jobs):
            continue
        first, second = [frames(Path(job['output'])) for job in jobs]
        if len(first) != 124 or len(second) != 124:
            raise ValueError('Incomplete generated segment')
        assembled = first[:120]+second[4:]
        folder = directory/observer
        output = folder/'enhanced_stitched.mp4'
        encode_frames(assembled,output,1344,768)
        paired(frames(folder/'source_24fps.mp4'),assembled,folder/'source_vs_enhanced.mp4')
        overlap_errors = []
        for index in range(8):
            a = np.asarray(Image.fromarray(first[116+index]).resize((336,192)),float)
            b = np.asarray(Image.fromarray(second[index]).resize((336,192)),float)
            overlap_errors.append(float(np.mean(np.abs(a-b))))
        canvas = Image.new('RGB',(1344,416),'#161616')
        draw = ImageDraw.Draw(canvas)
        for column,(rgb,label) in enumerate([(first[116],'Segment 1: overlap start'),(second[0],'Segment 2: same source time')]):
            canvas.paste(Image.fromarray(rgb).resize((672,384)),(column*672,32))
            draw.text((column*672+10,10),label,fill='white')
        canvas.save(folder/'overlap_comparison.jpg',quality=94)
        report = {'observer':observer, 'output':str(output), 'comparison':str(folder/'source_vs_enhanced.mp4'),
                  'frames':240,'fps':24,'duration_s':10.,'join_frame':120,
                  'overlap_same_time_rgb_mae_0_255':overlap_errors,
                  'seamless_continuity_certified':False,'strict_neural_pixel_gt':False,
                  'camera_pose_verdict':'not_certified'}
        save(folder/'assembly_report.json',report)
        products.append(report)
    report = {'phase':'complete' if all(s['phase']=='complete' for s in states.values()) else 'complete_with_failures',
              'jobs':states,'products':products,'cwm_lora_applied':False,
              'strict_neural_pixel_gt':False,'camera_pose_verdict':'not_certified'}
    save(directory/'suite_report.json',report)
    return report


def run(directory):
    manifest = read(directory/'suite_manifest.json')
    jobs = sorted(manifest['jobs'],key=lambda job:(job['id'][-2:],job['observer']))
    by_id = {job['id']:job for job in jobs}
    for job in jobs:
        if not Path(job['anchor']).is_file():
            raise FileNotFoundError(job['anchor'])
    states = {}
    for job in jobs:
        previous = read(Path(job['config']).parent/'status.json', {})
        states[job['id']] = {'phase': 'complete' if previous.get('phase') == 'complete' and Path(job['output']).is_file() else 'queued'}
    active = {}
    started = time.monotonic()
    status_path = directory/'batch_status.json'
    try:
        while True:
            for label,(child,gpu) in list(active.items()):
                work = Path(by_id[label]['config']).parent
                current = read(work/'status.json',{})
                states[label] = {'phase':current.get('phase','starting'), 'gpu':gpu,'pid':child.pid,
                                 'progress':read(work/'sampling_progress.json')}
                if child.poll() is not None:
                    states[label]['phase'] = 'complete' if child.returncode == 0 and current.get('phase')=='complete' else 'failed'
                    if states[label]['phase']=='failed':
                        states[label]['error'] = current.get('error',f'Exit {child.returncode}')
                    del active[label]
            if all(state['phase'] in ['complete','failed'] for state in states.values()):
                break
            if (directory/'STOP').exists():
                for child,_ in active.values():
                    child.terminate()
                save(status_path,{'phase':'stop_requested','jobs':states})
                return
            snapshot = gpu_snapshot()
            available = eligible_gpus(snapshot,{gpu for _,gpu in active.values()},
                                      manifest['allowed_gpus'],manifest['minimum_free_memory_mib'])
            for gpu in available:
                job = next((job for job in jobs if states[job['id']]['phase']=='queued'),None)
                if job is None:
                    break
                work = Path(job['config']).parent
                with (work/'inference.log').open('a') as log:
                    child = subprocess.Popen(module_command('inference.rgb',
                                              '--config',job['config']),cwd=ROOT,env=clean_environment(gpu,offline=True),
                                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[job['id']] = child,gpu
                states[job['id']] = {'phase':'starting','gpu':gpu,'pid':child.pid}
                print(json.dumps({'started':job['id'],'gpu':gpu,'pid':child.pid}),flush=True)
            save(status_path,{'phase':'running' if active else 'waiting_for_gpu','jobs':states,
                              'completed':sum(s['phase']=='complete' for s in states.values()),
                              'failed':sum(s['phase']=='failed' for s in states.values()),
                              'total':len(jobs),'elapsed_seconds':round(time.monotonic()-started),
                              'gpu_snapshot':snapshot,'cwm_lora_applied':False,'strict_neural_pixel_gt':False})
            if time.monotonic()-started>172800:
                raise TimeoutError('GPU queue exceeded 48 hours')
            time.sleep(15)
        save(status_path,{'phase':'exporting','jobs':states})
        report = export(directory,manifest,states)
        save(status_path,{'phase':report['phase'],'jobs':states,'report':str(directory/'suite_report.json')})
    except Exception as exc:
        save(status_path,{'phase':'failed','error':repr(exc),'jobs':states})
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    directory=parser.parse_args().directory.resolve()
    with (directory/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(directory)
