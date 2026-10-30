"""Depth + Semantic-ID to three-channel DUV conditioning."""

from __future__ import annotations

import math
from pathlib import Path

from .constants import (
    CONDITION_HEIGHT,
    CONDITION_WIDTH,
    DEPTH_FAR_METRES,
    DEPTH_NEAR_METRES,
    DEPTH_VALID_EPSILON_METRES,
    QWEN_OFFSETS,
    SEMANTIC_U,
    SEMANTIC_V,
    WINDOW_FRAMES,
)


def frame_paths(root: Path, ordinal: int) -> tuple[Path, Path]:
    return root / f"{ordinal:06d}.depth.f32", root / f"{ordinal:06d}.semantic_id.png"


def load_duv_frame(root: Path, ordinal: int, *, for_qwen: bool):
    import numpy as np
    from PIL import Image

    depth_path, semantic_path = frame_paths(root, ordinal)
    payload = depth_path.read_bytes()
    expected_bytes = CONDITION_WIDTH * CONDITION_HEIGHT * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"raw depth byte count must be {expected_bytes}: {depth_path}")
    metric = np.frombuffer(payload, dtype="<f4")
    if not bool(np.all(np.isfinite(metric))) or bool(np.any(metric < np.float32(0.0))):
        raise ValueError(f"raw depth must contain finite non-negative metres: {depth_path}")
    valid = metric > np.float32(DEPTH_VALID_EPSILON_METRES)
    codes = np.zeros(metric.shape, dtype=np.uint16)
    if bool(np.any(valid)):
        clipped = np.clip(metric[valid].astype(np.float64), DEPTH_NEAR_METRES, DEPTH_FAR_METRES)
        normalized = (math.log(DEPTH_FAR_METRES) - np.log(clipped)) / (
            math.log(DEPTH_FAR_METRES) - math.log(DEPTH_NEAR_METRES)
        )
        codes[valid] = np.floor(np.clip(normalized, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    depth = codes.reshape(CONDITION_HEIGHT, CONDITION_WIDTH)

    with Image.open(semantic_path) as image:
        if image.mode != "L" or image.size != (CONDITION_WIDTH, CONDITION_HEIGHT):
            raise ValueError(f"semantic ID must be L/{CONDITION_WIDTH}x{CONDITION_HEIGHT}: {semantic_path}")
        semantic = np.asarray(image, dtype=np.uint8).copy()
    if int(semantic.max()) > 11:
        raise ValueError(f"semantic ID must be in [0,11]: {semantic_path}")
    u = np.asarray(SEMANTIC_U, dtype=np.uint8)[semantic % 4]
    v = np.asarray(SEMANTIC_V, dtype=np.uint8)[semantic // 4]
    if for_qwen:
        depth8 = np.floor(depth.astype(np.float64) / 257.0 + 0.5).astype(np.uint8)
        return np.stack((depth8, u, v), axis=-1)
    return np.stack(
        (
            depth.astype(np.float32) / np.float32(65535.0),
            u.astype(np.float32) / np.float32(255.0),
            v.astype(np.float32) / np.float32(255.0),
        ),
        axis=-1,
    )


def load_duv_window(root: Path, start_frame: int, *, for_qwen: bool):
    import numpy as np

    offsets = QWEN_OFFSETS if for_qwen else range(WINDOW_FRAMES)
    return np.stack([load_duv_frame(root, start_frame + offset, for_qwen=for_qwen) for offset in offsets])
