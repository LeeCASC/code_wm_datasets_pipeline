import sys
from pathlib import Path
import unittest
import numpy as np
import torch

from world_model_pipeline.inference.rgb import adapt_video_decode_dtype
from world_model_pipeline.workflows.observers import selected_frames, scaled_intrinsics, camera_cv_m


class ObserverSegmentsTests(unittest.TestCase):
    def test_decode_adapter_uses_weight_dtype_without_mutating_float32_sample(self):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv3d(2,2,1).half()
            def decode(self, values):
                return self.layer(values)
        decoder = Decoder()
        latents = torch.ones(1,2,1,2,2,dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            decoder.decode(latents)
        result = adapt_video_decode_dtype(decoder).decode(latents)
        self.assertEqual(result.dtype,torch.float16)
        self.assertEqual(latents.dtype,torch.float32)

    def test_segment_coverage_and_source_pose_frame_mapping(self):
        indices = selected_frames()
        self.assertEqual(len(indices), 240)
        self.assertEqual(indices[116], 145)
        self.assertEqual(indices[-1], 299)
        timeline = np.arange(240)
        a, b = timeline[:124], timeline[116:]
        self.assertEqual(len(a), len(b))
        np.testing.assert_array_equal(a[116:], b[:8])
        np.testing.assert_array_equal(np.r_[a[:120], b[4:]], timeline)
        self.assertLessEqual(np.max(np.abs(indices/30-timeline/24)), 1/60+1e-10)

    def test_camera_conversion_units_and_local_axes(self):
        source = np.eye(4); source[:3,3] = [100,200,300]
        target = camera_cv_m(source)
        np.testing.assert_array_equal(target[:3,3], [1,-2,3])
        # UE forward maps to OpenCV forward; UE right maps to camera right.
        np.testing.assert_array_equal(target[:3,:3] @ [0,0,1], [1,0,0])
        np.testing.assert_array_equal(target[:3,:3] @ [1,0,0], [0,-1,0])
        self.assertAlmostEqual(np.linalg.det(target[:3,:3]), 1.)

    def test_resize_preserves_center_and_updates_both_focal_lengths(self):
        k = [[960,0,959.5],[0,960,539.5],[0,0,1]]
        resized = scaled_intrinsics(k, (1920,1080), (1344,768))
        self.assertAlmostEqual(resized[0,0], 672)
        self.assertAlmostEqual(resized[1,1], 960*768/1080)
        np.testing.assert_allclose(resized[:2,2], [671.5,383.5])


if __name__ == '__main__':
    unittest.main()
