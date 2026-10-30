"""Offline CWM worker supervisor. Model preparation is an explicit separate task."""
import argparse
import datetime
from pathlib import Path
import subprocess
import time

import av
import numpy as np

from world_model_pipeline.inference.cwm import runtime_config
from world_model_pipeline.io import read_json, write_json
from world_model_pipeline.paths import ROOT
from world_model_pipeline.runtime import clean_environment, module_command


def verify(config_path):
    config = runtime_config(config_path)
    stamps = []
    with av.open(str(config.output)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (1344, 768) or stream.average_rate != 24 or container.streams.audio:
            raise ValueError('Unexpected output video format')
        for index, frame in enumerate(container.decode(video=0)):
            stamps.append(float(frame.time))
            if index in {0, config.output_frames//2, config.output_frames-1}:
                frame.to_image().save(config_path.parent/f'generated_{index:06d}.png')
    if len(stamps) != config.output_frames or not np.allclose(np.diff(stamps), 1/24, atol=1e-6):
        raise ValueError('Output frame count or timestamps differ from the configured timeline')
    sources = [(config.condition_root.parent/'camera.json', 'camera_conditioning.json'),
               (config.condition_root.parent.parent/'flight_trajectory.json', 'flight_trajectory.json')]
    for source, name in sources:
        data = read_json(source)
        if data is None:
            continue
        data['frames'] = data['frames'][:config.output_frames]
        data['annotation_scope'] = 'Exact programmed proxy state/camera; generated pixel geometry is not certified.'
        write_json(config_path.parent/name, data)
    result = {'status': 'pass', 'output': str(config.output), 'decoded_frames': len(stamps),
              'fps': 24, 'resolution': [1344, 768], 'duration_s': len(stamps)/24,
              'audio_streams': 0, 'neural_video_generated': True, 'strict_neural_pixel_gt': False}
    write_json(config_path.parent/'video_validation.json', result)
    return result


def run(config_path, gpu):
    config_path = Path(config_path).resolve()
    folder = config_path.parent
    state = {'gpu': gpu, 'config': str(config_path), 'neural_video_generated': False,
             'strict_neural_pixel_gt': False, 'offline': True}
    started = time.monotonic()

    def status(phase, **values):
        state.update(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic()-started), **values)
        write_json(folder/'status.json', state)

    try:
        if runtime_config(config_path).output.exists():
            if read_json(folder/'status.json', {}).get('phase') == 'complete':
                return verify(config_path)
            raise FileExistsError(runtime_config(config_path).output)
        for stage in ('validate', 'prepare', 'generate'):
            status(stage)
            with (folder/f'{stage}.log').open('a') as log:
                child = subprocess.Popen(module_command('inference.cwm', stage, '--config', config_path),
                                         cwd=ROOT, env=clean_environment(gpu), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (folder/'STOP').exists():
                        child.terminate()
                        child.wait()
                        raise RuntimeError('Stopped by STOP file')
                    status(stage, child_pid=child.pid)
                    time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'{stage} failed; inspect {folder/f"{stage}.log"}')
        result = verify(config_path)
        status('complete', neural_video_generated=True, result=result)
        return result
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.gpu)
