import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from world_model_pipeline.geometry.aircraft import aircraft_mesh
from world_model_pipeline.geometry.obb import bounding_box_mesh
from world_model_pipeline.workflows.aircraft import SCENES
import world_model_pipeline.workflows.aircraft as prepare
from world_model_pipeline.orchestration.gpu import eligible_gpus


class PromptSuiteTests(unittest.TestCase):
    def test_changed_geometry_reuses_only_anchor_and_never_stale_text_or_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_cache = root/'source_cache'
            (source_cache/'window_000').mkdir(parents=True)
            (source_cache/'manifest.json').write_text(json.dumps({'status': 'complete'}))
            for name in ['anchor_reference.safetensors', 'input_video.safetensors',
                         'window_000/duv_reference.safetensors', 'window_000/qwen.safetensors']:
                (source_cache/name).write_bytes(name.encode())
            anchor = root/'anchor.png'; anchor.write_bytes(b'same appearance')
            source = SimpleNamespace(cache_dir=source_cache, models=SimpleNamespace(video_vae='same'),
                                     anchor_image=anchor, condition_root=root/'old_conditions')
            target = SimpleNamespace(cache_dir=root/'target_cache', models=source.models,
                                     anchor_image=anchor, condition_root=root/'box_conditions')
            target.cache_dir.mkdir()
            with patch.object(prepare, 'runtime_config', side_effect=[target, source]), \
                 patch('cwm_h3_inference.cache._begin_manifest'):
                prepare.seed_visual_cache(root/'target.json', root/'source.json')
            self.assertTrue((target.cache_dir/'anchor_reference.safetensors').exists())
            self.assertTrue((target.cache_dir/'input_video.safetensors').exists())
            self.assertFalse((target.cache_dir/'window_000/duv_reference.safetensors').exists())
            self.assertFalse((target.cache_dir/'window_000/qwen.safetensors').exists())
            self.assertEqual((source_cache/'window_000/qwen.safetensors').read_bytes(), b'window_000/qwen.safetensors')

    def test_single_cuboid_encloses_original_geometry_without_texture_or_aircraft_parts(self):
        plane = aircraft_mesh()
        box = bounding_box_mesh(plane)
        self.assertEqual(box.vertices.shape, (8, 3))
        self.assertEqual(box.faces.shape, (12, 3))
        np.testing.assert_array_equal(box.vertices.min(0), plane.vertices.min(0))
        np.testing.assert_array_equal(box.vertices.max(0), plane.vertices.max(0))
        self.assertEqual(len(np.unique(box.colors, axis=0)), 1)

    def test_gpu_queue_excludes_busy_in_use_and_unapproved_devices(self):
        snapshot = [{'index': 0, 'free_mib': 80000, 'utilization': 0},
                    {'index': 4, 'free_mib': 60000, 'utilization': 0},
                    {'index': 5, 'free_mib': 80000, 'utilization': 100},
                    {'index': 6, 'free_mib': 80000, 'utilization': 0},
                    {'index': 7, 'free_mib': 79000, 'utilization': 0}]
        self.assertEqual(eligible_gpus(snapshot, {6}, [4, 5, 6, 7], 77000), [7])

    def test_eight_distinct_scene_prompts_mark_anchor_conflicts(self):
        self.assertEqual(len(SCENES), 8)
        self.assertEqual(len({row[0] for row in SCENES}), 8)
        self.assertTrue(all(row[2] for row in SCENES[2:]))


if __name__ == '__main__':
    unittest.main()
