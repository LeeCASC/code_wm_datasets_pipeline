#!/usr/bin/env python3
"""Create eight text-only variations and two cuboid-input controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from world_model_pipeline.paths import ROOT
from world_model_pipeline.inference.cwm import runtime_config
from world_model_pipeline.geometry.obb import generate as generate_box
from world_model_pipeline.models import model_paths


SCENES = [
    ('01_terminal_day', 'Daylight terminal', False,
     'A large international airport in clear natural daylight. Beyond the active runway stand a long glass-and-steel passenger terminal, jet bridges, a distant air traffic control tower and two parked white passenger jets with blue tails. The terminal and parked aircraft remain well behind the hero airplane, separated from the active runway by airport grass and taxiways. Fine realistic airport infrastructure, crisp but natural atmospheric perspective.'),
    ('02_busy_apron', 'Apron and hangars', False,
     'An airport runway beside a busy but orderly distant apron. Three background passenger jets are parked at separate gates; low maintenance hangars, baggage carts, ground power units and a few service vehicles are visible near the terminal. All background aircraft and vehicles stay stationary during this shot, away from the active runway and never occluding the hero airplane. Natural midday daylight, photographic surface detail and a coherent airport layout.'),
    ('03_overcast', 'Overcast airport', True,
     'A realistic regional airport under a broad overcast sky with layered grey clouds. Soft diffuse daylight and subdued reflections on the airplane, muted green airport grass, a concrete control tower, low terminal buildings and one parked jet in the distance. The runway is dry, with restrained worn markings and believable surface texture. Flat airport terrain and no towering green walls.'),
    ('04_rain', 'Rain and wet runway', True,
     'A realistic airport in steady light rain under low grey clouds. The active asphalt runway is wet with thin puddles, subtle reflections of runway lights and a faint spray from the moving tires. Rain streaks and light mist remain restrained so the airplane and its landing gear stay clearly visible. A terminal, two parked jets and hangars appear behind the runway in a coherent distant airport layout. Wet grass, soft lighting and physically plausible reflections.'),
    ('05_fog', 'Morning fog', True,
     'An airport at early morning with low, gently drifting fog. The foreground airplane and nearby runway remain clearly readable; the distant terminal, air traffic control tower and a parked aircraft fade gradually into atmospheric haze. Pale cool illumination, slightly damp asphalt, subtle runway edge lights, realistic visibility falloff. Preserve the runway horizon and avoid an opaque white screen.'),
    ('06_sunset', 'Golden-hour airport', True,
     'A realistic airport at golden hour shortly before sunset. Warm low-angle sunlight creates long coherent shadows on the runway and soft reflections on the white fuselage. A distant passenger terminal, control tower and two parked jets are visible against a softly clouded orange-and-blue sky. Fine airport grass, believable atmospheric haze and restrained cinematic color grading without changing the camera framing.'),
    ('07_night', 'Night runway lights', True,
     'A realistic airport at night, illuminated by terminal windows, apron floodlights and a coherent line of runway edge lights. Blue taxiway lights and subtle red and white aircraft navigation lights are visible. The hero airplane remains readable under plausible airport lighting, with a dark blue sky and restrained reflections on asphalt. A lit passenger terminal, control tower and two parked jets occupy the far background; no daytime blue sky or bright sunlit grass.'),
    ('08_snow', 'Light snowfall', True,
     'A realistic airport during light snowfall under an overcast winter sky. The active runway is ploughed clear and remains dark damp asphalt, while thin snow covers the surrounding grass and distant airport rooftops. A terminal, hangar, control tower and two parked aircraft remain visible through gentle sparse snowflakes. Cool diffuse light, subtle tire spray and believable winter atmosphere, with the moving airplane clearly visible.'),
]

MOTION = ('Exactly one moving foreground airplane: the white twin-engine airplane with dark blue vertical tail from Picture 1. '
          'Preserve its identity, wings, engines and extended landing gear. From 0.00s to 3.50s it accelerates along the runway with its wheels on the surface; '
          'from 3.50s to 4.80s it gradually raises its nose; from 4.80s to 5.17s it starts to lift off while staying close to the runway. '
          'The foreground airplane position, occupied scale, body heading and pitch follow Video 1 frame by frame. '
          'The camera follows the supplied elevated left-side tracking viewpoint without cuts or additional zoom. '
          'Background objects stay beyond the active runway and never intersect the foreground airplane. '
          'Live-action aviation photography, fine natural materials, stable shadows and temporal continuity.')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def seed_visual_cache(config_path, source_config_path):
    """Copy only compatible visual encodings; never reuse text/Qwen/recovery."""
    from cwm_h3_inference.cache import _begin_manifest, cache_paths
    target, source = runtime_config(config_path), runtime_config(source_config_path)
    source_manifest = json.loads((source.cache_dir/'manifest.json').read_text())
    if source_manifest.get('status') != 'complete':
        raise ValueError('Visual cache source is not complete')
    if source.models.video_vae != target.models.video_vae:
        raise ValueError('Video VAE differs; cached visual encodings cannot be copied')
    same_anchor = hashlib.sha256(source.anchor_image.read_bytes()).digest() == hashlib.sha256(target.anchor_image.read_bytes()).digest()
    same_condition = source.condition_root == target.condition_root
    _begin_manifest(target)
    kinds = (['anchor', 'input_video'] if same_anchor else []) + (['reference'] if same_condition else [])
    receipt = []
    for kind in kinds:
        old, new = cache_paths(source, 0)[kind], cache_paths(target, 0)[kind]
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
        receipt.append({'kind': kind, 'source': str(old), 'target': str(new),
                        'sha256': hashlib.sha256(new.read_bytes()).hexdigest()})
    save(config_path.parent/'visual_cache_reuse.json', {'copied': receipt, 'qwen_reused': False, 'recovery_reused': False})


def build_config(proxy, anchor, directory, *, prompt, model_root=None):
    return {'models': model_paths(model_root), 'anchor_image': str(Path(anchor).resolve()),
            'condition_root': str(Path(proxy).resolve()/'side/condition'),
            'cache_dir': str(directory/'cache'), 'output': str(directory/'aircraft_takeoff.mp4'),
            'windows': [{'start_frame': 0, 'prompt': prompt}], 'seed': 1234,
            'steps': 20, 'attention_mode': 'flash3', 'blocks_to_swap': 48,
            'expected_lora_modules': 200}


def prepare_single(output, proxy, anchor, *, prompt=None, model_root=None):
    output, proxy, anchor = Path(output).resolve(), Path(proxy).resolve(), Path(anchor).resolve()
    if not anchor.is_file() or not (proxy/'side/condition').is_dir():
        raise FileNotFoundError('Provide an existing anchor image and a prepared proxy directory')
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(proxy, anchor, output, prompt=prompt or '[0.00s-5.17s] '+SCENES[0][3]+' '+MOTION,
                          model_root=model_root)
    save(output/'config.json', config)
    return {'config': str(output/'config.json')}


def prepare(output, proxy, anchor, *, model_root=None, reuse_from=None,
            allowed_gpus=(4, 5, 6, 7), minimum_free_memory_mib=77000):
    output = output.resolve()
    proxy, anchor = Path(proxy).resolve(), Path(anchor).resolve()
    if not anchor.is_file() or not (proxy/'side/condition').is_dir():
        raise FileNotFoundError('Provide an existing anchor image and a prepared proxy directory')
    output.mkdir(parents=True, exist_ok=False)
    box = output/'box_proxy'
    box_manifest = generate_box(proxy, box)
    jobs = []
    for label, title, conflict, description in SCENES:
        jobs.append({'id': label, 'title': title, 'prompt': '[0.00s-5.17s] '+description+' '+MOTION,
                     'geometry': 'coarse_aircraft', 'anchor': 'photoreal', 'anchor_conflict_expected': conflict})
    original = jobs[0]
    for label, title, anchor_kind in [('09_box_photo', 'Cuboid + photo anchor', 'photoreal'),
                                      ('10_box_only', 'Cuboid + cuboid anchor', 'box')]:
        jobs.append({'id': label, 'title': title, 'prompt': original['prompt'],
                     'geometry': 'single_cuboid', 'anchor': anchor_kind, 'anchor_conflict_expected': anchor_kind == 'box',
                     'paired_with': '01_terminal_day'})
    for job in jobs:
        run = output/'jobs'/job['id']
        run.mkdir(parents=True)
        config = build_config(proxy, anchor, run, prompt=job['prompt'], model_root=model_root)
        if job['geometry'] == 'single_cuboid':
            config['condition_root'] = str(box/'side/condition')
        if job['anchor'] == 'box':
            config['anchor_image'] = str(box/'side/anchor_box.png')
        path = run/'config.json'
        save(path, config)
        if reuse_from is not None:
            seed_visual_cache(path, Path(reuse_from).resolve())
        job.update(config=str(path), output=str(run/'aircraft_takeoff.a100_sdpa.mp4'))
        save(run/'experiment.json', job)
    manifest = {'schema': 'cwm_prompt_suite_v1', 'jobs': jobs, 'frames': 124, 'fps': 24, 'seed': 1234, 'steps': 20,
                'anchor_policy': 'Eight prompt tests hold the existing sunny photoreal anchor fixed; weather/night may conflict with its higher-priority appearance instructions.',
                'background_policy': 'Extra terminals and parked planes are requested only through text; they have no programmed 3D GT.',
                'box_geometry': box_manifest,
                'controls': '01 vs 09 changes only proxy geometry; 09 vs 10 changes only the RGB anchor. All share the same caption, camera matrices and body transforms.',
                'camera_pose_verdict': 'not_certified', 'strict_neural_pixel_gt': False,
                'allowed_gpus': list(allowed_gpus), 'minimum_free_memory_mib': minimum_free_memory_mib}
    save(output/'suite_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.proxy, args.anchor, model_root=args.model_root), indent=2))
