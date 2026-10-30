#!/usr/bin/env python3
"""Replace the aircraft with one untextured cuboid, preserving sampled poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from world_model_pipeline.paths import ROOT
from world_model_pipeline.geometry.aircraft import (Mesh, aircraft_mesh, audit, close_video, export_obj,
                            open_video, panel, render, write_json)


def bounding_box_mesh(original):
    low, high = original.vertices.min(0), original.vertices.max(0)
    vertices = np.array([[x, y, z] for x in [low[0], high[0]]
                         for y in [low[1], high[1]] for z in [low[2], high[2]]])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Mesh(vertices, faces, np.tile([178, 183, 190], (12, 1)).astype(np.uint8))


def generate(source, output, frame_count=124):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    side = output/'side'
    (side/'condition').mkdir(parents=True)
    scene = json.loads((source/'scene.json').read_text())
    trajectory = json.loads((source/'flight_trajectory.json').read_text())
    camera = json.loads((source/'side/camera.json').read_text())
    if frame_count > min(len(trajectory['frames']), len(camera['frames'])):
        raise ValueError('Source is shorter than the requested cuboid sequence')
    trajectory['frames'] = trajectory['frames'][:frame_count]
    camera['frames'] = camera['frames'][:frame_count]
    original = aircraft_mesh()
    box = bounding_box_mesh(original)
    export_obj(box, output/'aircraft_box.obj')
    write_json(output/'scene.json', scene)
    write_json(output/'flight_trajectory.json', trajectory)
    writer = open_video(side/'proxy.mp4', 672, 384, 24)
    try:
        for index, (state, cam) in enumerate(zip(trajectory['frames'], camera['frames'])):
            depth, sem, rgb = render(box, np.asarray(state['T_world_from_body']),
                                     np.asarray(cam['T_world_from_camera']), 336, 192,
                                     scene['horizontal_fov_deg'])
            depth.tofile(side/'condition'/f'{index:06d}.depth.f32')
            Image.fromarray(sem).save(side/'condition'/f'{index:06d}.semantic_id.png')
            cam['aircraft_visible_pixels'] = int((sem == 10).sum())
            writer.stdin.write(panel(rgb, 'single cuboid', index/24).tobytes())
            if index in {0, frame_count//2, frame_count-1}:
                Image.fromarray(rgb).resize((1344, 768), Image.Resampling.NEAREST).save(side/f'preview_{index:06d}.png')
            if index == 0:
                Image.fromarray(rgb).resize((1344, 768), Image.Resampling.NEAREST).save(side/'anchor_box.png')
    finally:
        close_video(writer)
    write_json(side/'camera.json', camera)
    manifest = {'status': 'proxy_generated', 'cwm_video_generated': False,
                'frame_count': frame_count, 'fps': 24, 'views': ['side'],
                'geometry': 'One solid body-aligned bounding cuboid; 8 vertices, 12 triangles; no textures.',
                'bounds_body_m': [box.vertices.min(0).tolist(), box.vertices.max(0).tolist()],
                'original_mesh_vertices': len(original.vertices), 'original_mesh_triangles': len(original.faces),
                'source': str(source), 'camera_and_body_poses_preserved': True,
                'limitation': 'A bounding volume contains empty space and can intersect the ground; it is not a physical aircraft surface.'}
    write_json(output/'manifest.json', manifest)
    write_json(output/'audit.json', audit(output))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=124)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output, args.frames), indent=2))
