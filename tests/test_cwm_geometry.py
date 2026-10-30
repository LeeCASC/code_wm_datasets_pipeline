"""Independent geometric checks for the planar diagnostic."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import world_model_pipeline.diagnostics.geometry as audit


class GeometryTests(unittest.TestCase):
    def test_plane_mapping_matches_independent_world_point_projection(self):
        k = np.array([[450., 0, 320], [0, 450, 180], [0, 0, 1]])
        a = np.eye(4); a[:3, 3] = [0, 0, 20]
        b = np.eye(4)
        angle = .13
        b[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        b[:3, 3] = [2., -1., 23.]
        xyz = np.array([[-3, 2, 0, 1], [4, -2, 0, 1], [.7, 1.2, 0, 1]])
        projections = []
        for camera in [a, b]:
            p = xyz @ camera[:3].T @ k.T
            projections.append(p[:, :2]/p[:, 2:3])
        predicted = audit.project_points(audit.ground_homography(k, a, b), projections[0])
        np.testing.assert_allclose(predicted, projections[1], atol=1e-10)

    def test_rgb_flow_detects_unprescribed_translation(self):
        cv2.setNumThreads(2)
        rng = np.random.default_rng(17)
        a = cv2.GaussianBlur(rng.integers(0, 256, (240, 400), dtype=np.uint8), (3, 3), .8)
        b = cv2.warpAffine(a, np.array([[1., 0, 7], [0, 1., 3]]), (400, 240))
        mask = np.zeros_like(a); mask[25:-25, 25:-25] = 255
        pa, pb, _ = audit.track_pair(a, b, mask, mask)
        self.assertGreater(len(pa), 80)
        observed = np.median(pb-pa, axis=0)
        np.testing.assert_allclose(observed, [7., 3.], atol=.2)
        # A falsely claimed stationary camera must have a large RGB residual.
        identity_prediction = audit.project_points(np.eye(3), pa)
        self.assertGreater(np.median(np.linalg.norm(pb-identity_prediction, axis=1)), 7.)


if __name__ == '__main__':
    unittest.main()
