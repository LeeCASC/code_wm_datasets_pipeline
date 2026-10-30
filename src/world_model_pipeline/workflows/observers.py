#!/usr/bin/env python3
"""Split two 30 FPS observer videos into overlapping 124-frame H3 references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import wave

import av
import numpy as np
from PIL import Image

from world_model_pipeline.paths import ROOT
from world_model_pipeline.models import model_paths
OBSERVERS = ['Observer07_OrbitRiseSlow', 'Observer08_OrbitFallSlow']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def selected_frames(count=240):
    # Nearest source frame, with explicit mapping retained for its actual pose.
    return np.floor(np.arange(count)*30/24+.5).astype(int)


def scaled_intrinsics(k, source_wh, target_wh):
    k = np.asarray(k, float).copy()
    sx, sy = np.asarray(target_wh)/np.asarray(source_wh)
    k[0, 0] *= sx; k[1, 1] *= sy
    k[0, 2] = (k[0, 2]+.5)*sx-.5
    k[1, 2] = (k[1, 2]+.5)*sy-.5
    return k


def camera_cv_m(camera_ue_cm):
    """Right-handed world (UE X,-Y,Z), OpenCV right/down/forward, in metres."""
    original = np.asarray(camera_ue_cm, float)
    world_change = np.diag([1., -1., 1.])
    cv_to_ue_local = np.array([[0., 0, 1], [1, 0, 0], [0, -1, 0]])
    result = np.eye(4)
    result[:3, :3] = world_change @ original[:3, :3] @ cv_to_ue_local
    result[:3, 3] = world_change @ original[:3, 3] / 100
    return result


def encode_frames(frames, output, width, height):
    with av.open(str(output), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'threads': '4'}
        for rgb in frames:
            image = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def caption(observer):
    motion = ('The camera slowly orbits around the foreground tanks while rising, matching every change of viewpoint in Video 1.'
              if 'Rise' in observer else
              'The camera slowly orbits around the foreground tanks while descending, matching every change of viewpoint in Video 1.')
    return ('[0.00s-5.17s] Re-render Video 1 as convincing live-action documentary footage of the same coastal training ground. '
            'Video 1 is the authoritative reference for camera movement, framing, object counts, geometry, foreground tank poses, '
            'their trajectories, occlusions, the containers, compound buildings, watchtowers, mountains and coastline. '
            'Picture 1 supplies the photographic materials, fine lighting and overall appearance; do not copy a static picture instead of following the video. '
            +motion+' Preserve the three prominent foreground tanks and all existing background objects without adding or removing foreground vehicles. '
            'Realistic matte painted metal, subtle weathering, dust on tracks, fine soil and wheel marks, restrained small stones and sparse low scrub. '
            'Natural daylight with soft sky fill, physically plausible ground bounce, stable contact shadows, restrained highlights and smooth tonal rolloff. '
            'Distant mountains have subtle aerial perspective; coastal water has natural small-scale reflections. '
            'Enrich fine background texture while preserving the scene layout and original occlusions. '
            'No new firing, explosions, dramatic action, camera cuts, artificial zoom or invented foreground structures. '
            'Maintain photographic realism and temporal consistency throughout the shot.')


def prepare(output, source_root, *, model_root=None, anchors_dir=None, observers=OBSERVERS,
            allowed_gpus=(4, 5, 6, 7), minimum_free_memory_mib=77000):
    output = output.resolve()
    source_root = Path(source_root).resolve()
    for observer in observers:
        if not (source_root/observer/'rgb/rgb.mp4').is_file():
            raise FileNotFoundError(source_root/observer/'rgb/rgb.mp4')
    output.mkdir(parents=True, exist_ok=False)
    assets = model_paths(model_root, cwm=False)
    silence = output/'schema_placeholder_silence.wav'
    with wave.open(str(silence), 'wb') as handle:
        handle.setnchannels(2); handle.setsampwidth(2); handle.setframerate(32000)
        handle.writeframes(np.zeros((165600, 2), dtype='<i2').tobytes())
    jobs = []
    indices = selected_frames()
    for observer in observers:
        source = source_root/observer
        rows = [json.loads(line) for line in (source/'frames.jsonl').read_text().splitlines()]
        meta = json.loads((source/'metadata.json').read_text())
        if len(rows) != 300 or meta['fps'] != 30:
            raise ValueError('Unexpected observer timebase')
        selected = {}
        with av.open(str(source/'rgb/rgb.mp4')) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in set(indices.tolist()):
                    selected[index] = frame.to_ndarray(format='rgb24')
        frames = [selected[int(index)] for index in indices]
        folder = output/observer
        folder.mkdir()
        encode_frames(frames, folder/'source_24fps.mp4', 1344, 768)
        for segment, start in enumerate([0, 116]):
            name = f'{observer}_seg{segment+1:02d}'
            run = output/'jobs'/name
            run.mkdir(parents=True)
            part = frames[start:start+124]
            if len(part) != 124:
                raise ValueError('Incomplete observer segment')
            encode_frames(part, run/'reference_672.mp4', 672, 384)
            encode_frames(part, run/'source_1344.mp4', 1344, 768)
            Image.fromarray(part[0]).save(run/'source_first.png')
            Image.fromarray(part[-1]).save(run/'source_last.png')
            if anchors_dir is not None:
                supplied = Path(anchors_dir)/f'{name}.png'
                if supplied.is_file():
                    shutil.copy2(supplied, run/'anchor_photoreal.png')
            mapped = []
            for local in range(124):
                global_frame = start+local
                source_index = int(indices[global_frame])
                row = rows[source_index]
                mapped.append({'segment_frame': local, 'timeline_frame_24fps': global_frame,
                               'segment_time_s': local/24,
                               'timeline_time_s': global_frame/24, 'source_frame': source_index,
                               'source_time_s': row['time_seconds'],
                               'camera_to_world_ue_cm': row['camera_to_world_ue'],
                               'T_world_from_camera_cv_m': camera_cv_m(row['camera_to_world_ue']).tolist()})
            pose = {'schema': 'observer_source_camera_mapping_v1', 'fps': 24, 'frames': mapped,
                    'K_output': scaled_intrinsics(meta['intrinsics']['matrix_pixels'], (1920,1080), (1344,768)).tolist(),
                    'K_rgb_reference': scaled_intrinsics(meta['intrinsics']['matrix_pixels'], (1920,1080), (672,384)).tolist(),
                    'world_convention': 'right-handed UE(X,-Y,Z), metres; OpenCV camera X-right,Y-down,Z-forward',
                    'annotation_scope': 'Poses belong to selected source RGB frames. Generated pixel camera geometry is not certified.',
                    'maximum_source_time_quantization_s': float(max(abs(r['source_time_s']-r['timeline_time_s']) for r in mapped))}
            save(run/'source_camera_mapping.json', pose)
            record = {'video_path': str(run/'reference_672.mp4'), 'audio_path': str(silence),
                      'caption': caption(observer), 'references': [
                          {'type': 'image', 'path': str(run/'anchor_photoreal.png')},
                          {'type': 'video', 'path': str(run/'reference_672.mp4')}]}
            (run/'references.jsonl').write_text(json.dumps(record)+'\n')
            config = {'id': name, 'observer': observer, 'segment_index': segment,
                      'start_frame_24fps': start, 'frame_count': 124, 'fps': 24,
                      'models': assets, 'references_jsonl': str(run/'references.jsonl'),
                      'source_reference': str(run/'reference_672.mp4'), 'source_comparison': str(run/'source_1344.mp4'),
                      'anchor': str(run/'anchor_photoreal.png'), 'seed': 1234, 'steps': 20,
                      'width': 1344, 'height': 768, 'blocks_to_swap': 48,
                      'output': str(run/'enhanced.mp4'), 'native_joint_output': str(run/'native_joint_av.mp4'),
                      'cwm_lora_applied': False, 'camera_pose_verdict': 'not_certified',
                      'placeholder_audio_note': 'Silent WAV satisfies the unified JSONL record schema; it is not a reference condition.'}
            save(run/'config.json', config)
            jobs.append({'id': name, 'observer': observer, 'config': str(run/'config.json'),
                         'source_first': str(run/'source_first.png'), 'anchor': str(run/'anchor_photoreal.png'),
                         'output': str(run/'enhanced.mp4')})
        del frames, selected
    manifest = {'schema': 'observer_segmented_ref2va_v1', 'jobs': jobs, 'allowed_gpus': list(allowed_gpus),
                'minimum_free_memory_mib': minimum_free_memory_mib, 'output_fps': 24, 'original_frames': 300,
                'resampled_frames': 240, 'segment_starts': [0,116], 'segment_frames': 124,
                'overlap_frames': 8, 'assembly': 'First segment frames [0:120], second [4:124]; no crossfade.',
                'method': 'Native unmerged H3 Ref2VA; RGB video reference, photoreal image reference and text.',
                'cwm_lora_applied': False, 'strict_neural_pixel_gt': False,
                'continuity': 'Independent segments; matching style and overlap do not guarantee seamless continuity.'}
    save(output/'suite_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--anchors-dir', type=Path)
    parser.add_argument('--model-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.source, anchors_dir=args.anchors_dir,
                             model_root=args.model_root), indent=2))
