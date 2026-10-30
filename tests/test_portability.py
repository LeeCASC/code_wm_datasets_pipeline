import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import av
import numpy as np
import pytest

from world_model_pipeline.paths import ROOT
from world_model_pipeline.runtime import clean_environment
from world_model_pipeline.video import concatenate
from world_model_pipeline.workflows.aircraft import prepare_single, prepare


def tiny_video(path, count, color):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        stream.options = {'crf': '18', 'threads': '1'}
        for i in range(count):
            rgb = np.full((48, 64, 3), color+i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_cli_runs_outside_checkout_without_pythonpath(tmp_path):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/wmp'), '--help'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert 'aircraft' in result.stdout and 'ue' in result.stdout


def test_dependencies_use_vendored_sources():
    import cwm_h3_inference
    import musubi_tuner
    assert Path(cwm_h3_inference.__file__).is_relative_to(ROOT/'third_party/code_world_model')
    assert Path(musubi_tuner.__file__).is_relative_to(ROOT/'third_party/musubi_tuner')


def test_model_configs_use_new_repository_and_not_old_output(tmp_path):
    proxy = tmp_path/'proxy'
    (proxy/'side/condition').mkdir(parents=True)
    anchor = ROOT/'assets/examples/aircraft/anchor_photoreal.png'
    result = prepare_single(tmp_path/'job', proxy, anchor)
    value = json.loads(Path(result['config']).read_text())
    assert value['models']['dit'] == str(ROOT/'models/diffusion_models/minimax_h3_ref2va_bf16.safetensors')
    assert value['condition_root'] == str(proxy/'side/condition')
    assert '20261029' not in json.dumps(value)


def test_runtime_removes_proxies_and_keeps_subprocess_importable():
    with patch.dict(os.environ, {'HTTPS_PROXY': 'http://127.0.0.1:9', 'ALL_PROXY': 'http://127.0.0.1:9'}):
        env = clean_environment(4)
    assert 'HTTPS_PROXY' not in env and 'ALL_PROXY' not in env
    assert env['CUDA_VISIBLE_DEVICES'] == '4'
    assert env['PYTHONPATH'] == str(ROOT/'src')
    assert env['HF_HUB_OFFLINE'] == '1'


def test_all_suite_jobs_keep_requested_anchor_except_explicit_box_control(tmp_path):
    proxy = tmp_path/'proxy'
    (proxy/'side/condition').mkdir(parents=True)
    anchor = ROOT/'assets/examples/aircraft/anchor_photoreal.png'

    def box_fixture(source, output):
        (output/'side/condition').mkdir(parents=True)
        (output/'side/anchor_box.png').write_bytes(b'box reference')
        return {'status': 'proxy_generated'}

    with patch('world_model_pipeline.workflows.aircraft.generate_box', side_effect=box_fixture):
        manifest = prepare(tmp_path/'suite', proxy, anchor)
    assert len(manifest['jobs']) == 10
    for job in manifest['jobs']:
        config = json.loads(Path(job['config']).read_text())
        expected = tmp_path/'suite/box_proxy/side/anchor_box.png' if job['id']=='10_box_only' else anchor
        assert Path(config['anchor_image']) == expected
        assert Path(config['anchor_image']).is_file()


def test_concat_preserves_all_frames_timestamps_and_boundary_pixels(tmp_path):
    a, b, out = tmp_path/'first clip.mp4', tmp_path/'second clip.mp4', tmp_path/'joined.mp4'
    tiny_video(a, 4, 10)
    tiny_video(b, 6, 100)
    report = concatenate([a, b], out)
    assert report['frames'] == 10
    assert report['source_boundary_hashes_matched'] == 4
    assert report['sources'][1]['start_frame'] == 4
    assert report['stream_copy']
    with pytest.raises(FileExistsError):
        concatenate([a, b], out)
