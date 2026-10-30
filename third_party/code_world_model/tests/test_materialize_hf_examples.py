from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_hf_examples.py"
SPEC = importlib.util.spec_from_file_location("materialize_hf_examples", SCRIPT)
assert SPEC and SPEC.loader
materialize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materialize)


def test_fixed_log_inverse_round_trips_every_uint16_code() -> None:
    codes = np.arange(65536, dtype=np.uint16)
    metric = materialize.fixed_log_codes_to_metric(codes)

    assert metric.dtype == np.dtype("float32")
    assert np.array_equal(materialize.metric_to_fixed_log_codes(metric), codes)


def _fixture(tmp_path: Path, *, depth_format: str) -> tuple[Path, Path]:
    source = tmp_path / "compact"
    example = source / "1"
    (example / "condition").mkdir(parents=True)
    (example / "prompts").mkdir()
    frames = 124
    if depth_format == materialize.FIXED_LOG_DEPTH:
        depth = np.arange(frames * materialize.HEIGHT * materialize.WIDTH, dtype=np.uint16).reshape(
            frames, materialize.HEIGHT, materialize.WIDTH
        )
    else:
        depth = np.linspace(0.0, 256.0, frames * materialize.HEIGHT * materialize.WIDTH, dtype=np.float32).reshape(
            frames, materialize.HEIGHT, materialize.WIDTH
        )
    semantic = np.arange(frames * materialize.HEIGHT * materialize.WIDTH, dtype=np.uint8).reshape(
        frames, materialize.HEIGHT, materialize.WIDTH
    ) % 12
    np.savez_compressed(example / "condition" / "000.npz", depth=depth, semantic_id=semantic)
    Image.new("RGB", (1536, 864), (1, 2, 3)).save(example / "first_frame.png")
    prompt = "[0.00s-5.17s] exact prompt"
    (example / "prompts" / "000.txt").write_text(prompt + "\n", encoding="utf-8")
    metadata = {
        "id": 1,
        "fps": 24,
        "frames": frames,
        "resolution": [materialize.WIDTH, materialize.HEIGHT],
        "depth_format": depth_format,
        "first_frame": "first_frame.png",
        "prompts": [{
            "index": 0,
            "file": "prompts/000.txt",
            "conditioning_frame_start": 0,
            "conditioning_frame_end_exclusive": 124,
        }],
        "condition_chunks": [{
            "file": "condition/000.npz",
            "start_frame": 0,
            "end_frame_exclusive": 124,
            "frames": 124,
        }],
    }
    (example / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps({"schema": "code_world_model_inference_examples_v3", "count": 40, "expected_count": 40}),
        encoding="utf-8",
    )
    output = tmp_path / "release"
    destination = output / "examples" / "example_01"
    destination.mkdir(parents=True)
    config = {
        "models": materialize.RELEASE_MODELS,
        "anchor_image": "first_frame.png",
        "condition_root": "conditions",
        "windows": [{"prompt": prompt, "start_frame": 0}],
        "seed": materialize.RELEASE_SEEDS[1],
        "steps": 20,
        "attention_mode": "flash3",
        "blocks_to_swap": 48,
        "expected_lora_modules": 200,
        "cache_dir": "../../outputs/example_01/cache",
        "output": "../../outputs/example_01/output.mp4",
    }
    (destination / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return source, output


def test_materializes_metric_example(tmp_path: Path) -> None:
    source, output = _fixture(tmp_path, depth_format=materialize.METRIC_DEPTH)

    result = materialize.materialize_example(source, output, 1, require_config=True, frame_limit=2)

    assert result["materialized_frames"] == 2
    depth = np.fromfile(output / "examples/example_01/conditions/000000.depth.f32", dtype="<f4")
    assert depth.size == materialize.WIDTH * materialize.HEIGHT
    with Image.open(output / "examples/example_01/conditions/000000.semantic_id.png") as image:
        assert image.mode == "L"
        assert image.size == (materialize.WIDTH, materialize.HEIGHT)


def test_creates_released_config_from_public_example(tmp_path: Path) -> None:
    source, output = _fixture(tmp_path, depth_format=materialize.METRIC_DEPTH)
    config_path = output / "examples/example_01/config.json"
    config_path.unlink()

    materialize.materialize_example(source, output, 1, require_config=True, frame_limit=1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["seed"] == materialize.RELEASE_SEEDS[1]
    assert config["windows"] == [{"prompt": "[0.00s-5.17s] exact prompt", "start_frame": 0}]
    assert config["models"] == materialize.RELEASE_MODELS


def test_materializes_fixed_log_example_without_changing_codes(tmp_path: Path) -> None:
    source, output = _fixture(tmp_path, depth_format=materialize.FIXED_LOG_DEPTH)

    materialize.materialize_example(source, output, 1, require_config=True, frame_limit=2)

    with np.load(source / "1/condition/000.npz", allow_pickle=False) as payload:
        expected = payload["depth"][0]
    metric = np.fromfile(
        output / "examples/example_01/conditions/000000.depth.f32", dtype="<f4"
    ).reshape(materialize.HEIGHT, materialize.WIDTH)
    assert np.array_equal(materialize.metric_to_fixed_log_codes(metric), expected)


def test_rejects_prompt_drift(tmp_path: Path) -> None:
    source, output = _fixture(tmp_path, depth_format=materialize.METRIC_DEPTH)
    config_path = output / "examples/example_01/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["windows"][0]["prompt"] = "changed"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        materialize.materialize_example(source, output, 1, require_config=True, frame_limit=1)
    except ValueError as error:
        assert "config prompt differs" in str(error)
    else:
        raise AssertionError("prompt drift was accepted")
