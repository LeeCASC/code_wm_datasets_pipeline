from __future__ import annotations

import math

import numpy as np
from PIL import Image
import pytest

from cwm_h3_inference.constants import CONDITION_HEIGHT, CONDITION_WIDTH
from cwm_h3_inference.duv import load_duv_frame


def _write_frame(root, depth: np.ndarray, semantic: np.ndarray) -> None:
    root.mkdir()
    (root / "000000.depth.f32").write_bytes(np.asarray(depth, dtype="<f4").tobytes())
    Image.fromarray(np.asarray(semantic, dtype=np.uint8), mode="L").save(
        root / "000000.semantic_id.png"
    )


def _valid_arrays() -> tuple[np.ndarray, np.ndarray]:
    depth = np.full((CONDITION_HEIGHT, CONDITION_WIDTH), 1.0, dtype=np.float32)
    depth[0, :4] = (0.0, 0.3, 256.0, 512.0)
    semantic = np.indices((CONDITION_HEIGHT, CONDITION_WIDTH))[1] % 12
    return depth, semantic.astype(np.uint8)


def test_duv_conversion_for_qwen_and_vae(tmp_path) -> None:
    depth, semantic = _valid_arrays()
    _write_frame(tmp_path / "condition", depth, semantic)

    qwen = load_duv_frame(tmp_path / "condition", 0, for_qwen=True)
    vae = load_duv_frame(tmp_path / "condition", 0, for_qwen=False)

    expected_one_metre = round(
        (math.log(256.0) - math.log(1.0))
        / (math.log(256.0) - math.log(0.3))
        * 65535.0
    )
    assert qwen.shape == (CONDITION_HEIGHT, CONDITION_WIDTH, 3)
    assert qwen.dtype == np.uint8
    assert qwen[0, :4, 0].tolist() == [0, 255, 0, 0]
    assert int(qwen[1, 0, 0]) == round(expected_one_metre / 257.0)
    assert qwen[0, 0].tolist() == [0, 32, 43]
    assert qwen[0, 4, 1:].tolist() == [32, 128]
    assert qwen[0, 8, 1:].tolist() == [32, 213]

    assert vae.shape == (CONDITION_HEIGHT, CONDITION_WIDTH, 3)
    assert vae.dtype == np.float32
    assert vae[0, 0, 0] == pytest.approx(0.0)
    assert vae[0, 1, 0] == pytest.approx(1.0)
    assert vae[0, 4, 1] == pytest.approx(32 / 255)
    assert vae[0, 8, 2] == pytest.approx(213 / 255)


@pytest.mark.parametrize("bad_value", [-0.01, np.nan, np.inf])
def test_duv_rejects_invalid_raw_depth_values(tmp_path, bad_value) -> None:
    depth, semantic = _valid_arrays()
    depth[0, 0] = bad_value
    _write_frame(tmp_path / "condition", depth, semantic)

    with pytest.raises(ValueError, match="finite non-negative"):
        load_duv_frame(tmp_path / "condition", 0, for_qwen=False)


def test_duv_rejects_out_of_range_semantic_id(tmp_path) -> None:
    depth, semantic = _valid_arrays()
    semantic[0, 0] = 12
    _write_frame(tmp_path / "condition", depth, semantic)

    with pytest.raises(ValueError, match=r"\[0,11\]"):
        load_duv_frame(tmp_path / "condition", 0, for_qwen=True)


def test_duv_rejects_wrong_raw_depth_size(tmp_path) -> None:
    depth, semantic = _valid_arrays()
    _write_frame(tmp_path / "condition", depth[:-1], semantic)

    with pytest.raises(ValueError, match="raw depth byte count"):
        load_duv_frame(tmp_path / "condition", 0, for_qwen=False)
