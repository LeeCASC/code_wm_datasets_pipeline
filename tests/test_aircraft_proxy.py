import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from world_model_pipeline.geometry.aircraft import (Scene, aircraft_mesh, camera_for, camera_matrix,
                            clip_near, intrinsics, raster_triangle, render,
                            world_state, cwm_template, generate)


class ProjectionTests(unittest.TestCase):
    def test_ground_depth_unprojects_to_ground_and_is_not_ray_range(self):
        mesh = aircraft_mesh()
        transform = np.eye(4)
        transform[:3, 3] = [10000, 0, 10000]  # No aircraft in this camera.
        camera = camera_matrix([0, -20, 10], [0, 0, 0])
        depth, semantic, _ = render(mesh, transform, camera, 80, 48, 58)
        yy, xx = np.nonzero((semantic == 3) | (semantic == 4))
        k = intrinsics(80, 48, 58)
        rays = np.stack([(xx+.5-k[0, 2])/k[0, 0],
                         (yy+.5-k[1, 2])/k[1, 1], np.ones(len(xx))], axis=1)
        xyz = (rays*depth[yy, xx, None]) @ camera[:3, :3].T + camera[:3, 3]
        np.testing.assert_allclose(xyz[:, 2], 0, atol=2e-6)
        self.assertGreater(np.max(np.linalg.norm(rays, axis=1)-1), .05)

    def test_perspective_depth_uses_reciprocal_interpolation(self):
        # Projected vertices at (0,0), (4,0), (0,4); z values 1,2,4.
        triangle = np.array([[0, 0, 1.], [8, 0, 2.], [0, 16, 4.]])
        depth = np.full((4, 4), np.inf)
        semantic = np.zeros((4, 4), dtype=np.uint8)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        raster_triangle(triangle, [1, 2, 3], depth, semantic, rgb, np.eye(3))
        # Pixel center (.5,.5): weights .75,.125,.125.
        self.assertAlmostEqual(depth[0, 0], 1/(.75+.125/2+.125/4))
        self.assertEqual(semantic[0, 0], 10)
        # A farther triangle must be occluded.
        raster_triangle(triangle*2, [9, 9, 9], depth, semantic, rgb, np.eye(3))
        np.testing.assert_array_equal(rgb[0, 0], [1, 2, 3])

    def test_near_plane_clipping_handles_crossing_and_hidden_triangles(self):
        triangles = clip_near(np.array([[0, 0, -.1], [1, 0, 1.], [0, 1, 1.]]))
        self.assertEqual(len(triangles), 2)
        self.assertTrue(all((tri[:, 2] >= .05-1e-12).all() for tri in triangles))
        self.assertEqual(clip_near(np.array([[0, 0, -1.], [1, 0, -1.], [0, 1, -1.]])), [])

    def test_all_vertices_remain_inside_every_camera_during_takeoff(self):
        scene, mesh = Scene(), aircraft_mesh()
        for frame in range(scene.frames):
            transform, _, _, _ = world_state(scene, frame/scene.fps, mesh)
            world = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
            for view in ('side', 'rear', 'fixed'):
                camera = camera_for(view, transform[:3, 3])
                c = (world-camera[:3, 3]) @ camera[:3, :3]
                self.assertTrue((c[:, 2] > .05).all())
                self.assertTrue((c[:, 2] < 256).all(), (view, frame, c[:, 2].max()))
                k = intrinsics(336, 192, scene.horizontal_fov_deg)
                xy = c[:, :2]/c[:, 2:3] * k[0, 0] + k[:2, 2]
                self.assertTrue(((xy > [2, 2]) & (xy < [334, 190])).all(), (view, frame, xy.min(0), xy.max(0)))


class WorldTests(unittest.TestCase):
    def test_motion_accelerates_ground_contact_and_positive_pitch(self):
        scene, mesh = Scene(), aircraft_mesh()
        previous_x, previous_dx = None, None
        for frame in range(scene.frames):
            t = frame/scene.fps
            transform, pitch, clearance, _ = world_state(scene, t, mesh)
            rotation = transform[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            world = mesh.vertices @ rotation.T + transform[:3, 3]
            self.assertAlmostEqual(world[:, 2].min(), clearance, places=10)
            if t <= scene.liftoff_s:
                self.assertEqual(clearance, 0)
            if pitch > 0:
                self.assertGreater((rotation @ [1, 0, 0])[2], 0)
            x = transform[0, 3]
            if previous_x is not None:
                dx = x-previous_x
                if previous_dx is not None:
                    self.assertGreater(dx, previous_dx)
                previous_dx = dx
            previous_x = x

    def test_fixed_camera_is_fixed_while_tracking_camera_moves(self):
        np.testing.assert_array_equal(camera_for('fixed', [0,0,2]), camera_for('fixed', [230,0,17]))
        self.assertFalse(np.allclose(camera_for('side', [0,0,2]), camera_for('side', [230,0,17])))

    def test_generated_config_matches_actual_upstream_parser(self):
        from cwm_h3_inference.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cwm.json'
            path.write_text(json.dumps(cwm_template(Path(tmp), Scene(), 'side')))
            config = load_config(path, check_files=False)
            self.assertEqual(config.output_frames, 214)
            self.assertEqual([w.start_frame for w in config.windows], [0, 90])
            self.assertEqual(config.attention_mode, 'flash3')

    def test_existing_output_refused_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'keep.txt'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                generate(tmp, Scene())
            self.assertEqual(marker.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
