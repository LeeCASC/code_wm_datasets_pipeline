#!/usr/bin/env python3
"""Expand the published compact CWM examples into inference-ready inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image


WIDTH = 336
HEIGHT = 192
WINDOW_FRAMES = 124
STRIDE_FRAMES = 90
NEAR_METRES = 0.3
FAR_METRES = 256.0
METRIC_DEPTH = "float32_le_metric_first_surface_metres"
FIXED_LOG_DEPTH = "uint16_le_fixed_log_gray16_codes"
SUPPORTED_DEPTH_FORMATS = {METRIC_DEPTH, FIXED_LOG_DEPTH}
RELEASE_MODELS = {
    "audio_vae": "../../models/vae/minimax_h3_audio_vae_fp32.safetensors",
    "dit": "../../models/diffusion_models/minimax_h3_ref2va_bf16.safetensors",
    "lora": "../../models/lora/minimax-h3-cwm.safetensors",
    "processor": "../../models/processors/Qwen3-VL-32B-Instruct",
    "text_encoder": "../../models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    "video_vae": "../../models/vae/minimax_h3_video_vae_fp16.safetensors",
}
RELEASE_SEEDS = {
    1: 9942001,
    2: 9800002,
    3: 9700005,
    4: 9700006,
    5: 9700007,
    6: 9700013,
    7: 9700020,
    8: 9700021,
    9: 9700030,
    10: 9700036,
    11: 9700037,
    12: 9700040,
    13: 9700044,
    14: 9700045,
    15: 9700046,
    16: 9700047,
    17: 9700051,
    18: 9700054,
    19: 9700055,
    20: 9700057,
    21: 9700063,
    22: 9700065,
    23: 9700069,
    24: 9700071,
    25: 9700078,
    26: 9700088,
    27: 9700094,
    28: 9600014,
    29: 9600020,
    30: 9600023,
    31: 9600028,
    32: 9600040,
    33: 9600042,
    34: 9300038,
    35: 9400059,
    36: 9300066,
    37: 9300069,
    38: 9800006,
    39: 9800010,
    40: 9800013,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_prompt(value: str) -> str:
    return value.strip().replace("\r\n", "\n").replace("\r", "\n")


def build_release_config(
    example_root: Path, metadata: dict[str, Any], example_id: int
) -> dict[str, Any]:
    windows = []
    for prompt_item in metadata["prompts"]:
        prompt = _canonical_prompt(
            (example_root / prompt_item["file"]).read_text(encoding="utf-8")
        )
        # Preserve the text representation used by the released configurations.
        if example_id != 2:
            prompt = prompt.replace("\n", "\r\n")
        windows.append(
            {
                "prompt": prompt,
                "start_frame": prompt_item["conditioning_frame_start"],
            }
        )
    name = f"example_{example_id:02d}"
    return {
        "anchor_image": "first_frame.png",
        "attention_mode": "flash3",
        "blocks_to_swap": 48,
        "cache_dir": f"../../outputs/{name}/cache",
        "condition_root": "conditions",
        "expected_lora_modules": 200,
        "models": RELEASE_MODELS,
        "output": f"../../outputs/{name}/output.mp4",
        "seed": RELEASE_SEEDS[example_id],
        "steps": 20,
        "windows": windows,
    }


def ensure_release_config(
    config_path: Path, example_root: Path, metadata: dict[str, Any], example_id: int
) -> None:
    if not config_path.exists():
        payload = json.dumps(
            build_release_config(example_root, metadata, example_id),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        _atomic_bytes(config_path, (payload + "\n").encode("utf-8"))
    validate_config(config_path, example_root, metadata, example_id)


def fixed_log_codes_to_metric(codes: np.ndarray) -> np.ndarray:
    """Choose float32 depths that round-trip to every published uint16 code."""
    require(codes.dtype == np.dtype("uint16"), f"fixed-log depth must be uint16, got {codes.dtype}")
    metric = np.zeros(codes.shape, dtype="<f4")
    valid = codes > 0
    if bool(np.any(valid)):
        log_near = math.log(NEAR_METRES)
        log_far = math.log(FAR_METRES)
        normalized = codes[valid].astype(np.float64) / 65535.0
        metric[valid] = np.exp(log_far - normalized * (log_far - log_near)).astype(np.float32)
    return np.ascontiguousarray(metric)


def metric_to_fixed_log_codes(metric: np.ndarray) -> np.ndarray:
    """Reference implementation matching the inference loader."""
    metric = np.asarray(metric, dtype=np.float32)
    valid = metric > np.float32(0.001)
    codes = np.zeros(metric.shape, dtype=np.uint16)
    if bool(np.any(valid)):
        clipped = np.clip(metric[valid].astype(np.float64), NEAR_METRES, FAR_METRES)
        normalized = (math.log(FAR_METRES) - np.log(clipped)) / (
            math.log(FAR_METRES) - math.log(NEAR_METRES)
        )
        codes[valid] = np.floor(np.clip(normalized, 0.0, 1.0) * 65535.0 + 0.5).astype(
            np.uint16
        )
    return codes


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"existing file differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_semantic_png(path: Path, semantic_id: np.ndarray) -> None:
    semantic_id = np.ascontiguousarray(semantic_id, dtype=np.uint8)
    if path.exists():
        with Image.open(path) as image:
            existing = np.asarray(image.convert("L"), dtype=np.uint8)
        require(np.array_equal(existing, semantic_id), f"existing Semantic-ID differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(semantic_id, mode="L").save(temporary, format="PNG", compress_level=6)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_exact(source: Path, destination: Path) -> None:
    require(source.is_file(), f"missing source file: {source}")
    payload = source.read_bytes()
    _atomic_bytes(destination, payload)


def validate_metadata(example_root: Path, metadata: dict[str, Any], example_id: int) -> None:
    require(metadata.get("id") == example_id, f"example {example_id}: metadata id mismatch")
    require(metadata.get("fps") == 24, f"example {example_id}: fps must be 24")
    require(metadata.get("resolution") == [WIDTH, HEIGHT], f"example {example_id}: bad resolution")
    require(metadata.get("depth_format") in SUPPORTED_DEPTH_FORMATS, f"example {example_id}: unsupported depth format")
    frames = metadata.get("frames")
    require(type(frames) is int and frames >= WINDOW_FRAMES, f"example {example_id}: invalid frame count")
    prompts = metadata.get("prompts")
    require(isinstance(prompts, list) and prompts, f"example {example_id}: prompts missing")
    require(frames == WINDOW_FRAMES + STRIDE_FRAMES * (len(prompts) - 1), f"example {example_id}: frame/window mismatch")
    for index, item in enumerate(prompts):
        require(item.get("index") == index, f"example {example_id}: prompt index mismatch")
        require(item.get("conditioning_frame_start") == index * STRIDE_FRAMES, f"example {example_id}: prompt start mismatch")
        require(item.get("conditioning_frame_end_exclusive") == index * STRIDE_FRAMES + WINDOW_FRAMES, f"example {example_id}: prompt end mismatch")
        prompt_path = example_root / item.get("file", "")
        require(prompt_path.is_file(), f"example {example_id}: missing prompt {prompt_path}")
    chunks = metadata.get("condition_chunks")
    require(isinstance(chunks, list) and chunks, f"example {example_id}: condition chunks missing")
    cursor = 0
    for item in chunks:
        start = item.get("start_frame")
        end = item.get("end_frame_exclusive")
        require(start == cursor and type(end) is int and end > start, f"example {example_id}: non-contiguous chunks")
        require(item.get("frames") == end - start, f"example {example_id}: chunk frame count mismatch")
        chunk_path = example_root / item.get("file", "")
        require(chunk_path.is_file(), f"example {example_id}: missing chunk {chunk_path}")
        cursor = end
    require(cursor == frames, f"example {example_id}: chunks do not cover all frames")


def validate_config(config_path: Path, example_root: Path, metadata: dict[str, Any], example_id: int) -> None:
    require(config_path.is_file(), f"missing released config: {config_path}")
    config = _read_json(config_path)
    require(config.get("anchor_image") == "first_frame.png", f"example {example_id}: config anchor path mismatch")
    require(config.get("condition_root") == "conditions", f"example {example_id}: config condition path mismatch")
    require(config.get("seed") == RELEASE_SEEDS[example_id], f"example {example_id}: released seed mismatch")
    require(config.get("steps") == 20, f"example {example_id}: steps mismatch")
    require(config.get("attention_mode") == "flash3", f"example {example_id}: attention mode mismatch")
    require(config.get("blocks_to_swap") == 48, f"example {example_id}: block swap count mismatch")
    require(config.get("expected_lora_modules") == 200, f"example {example_id}: LoRA module count mismatch")
    require(config.get("models") == RELEASE_MODELS, f"example {example_id}: model paths mismatch")
    name = f"example_{example_id:02d}"
    require(config.get("cache_dir") == f"../../outputs/{name}/cache", f"example {example_id}: cache path mismatch")
    require(config.get("output") == f"../../outputs/{name}/output.mp4", f"example {example_id}: output path mismatch")
    windows = config.get("windows")
    prompt_items = metadata["prompts"]
    require(isinstance(windows, list) and len(windows) == len(prompt_items), f"example {example_id}: config window count mismatch")
    for window, prompt_item in zip(windows, prompt_items, strict=True):
        require(window.get("start_frame") == prompt_item["conditioning_frame_start"], f"example {example_id}: config window start mismatch")
        published = (example_root / prompt_item["file"]).read_text(encoding="utf-8")
        require(_canonical_prompt(window.get("prompt", "")) == _canonical_prompt(published), f"example {example_id}: config prompt differs from published prompt")


def _iter_chunks(example_root: Path, metadata: dict[str, Any]) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    depth_format = metadata["depth_format"]
    for chunk in metadata["condition_chunks"]:
        path = example_root / chunk["file"]
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == {"depth", "semantic_id"}, f"unexpected arrays in {path}")
            depth = payload["depth"]
            semantic_id = payload["semantic_id"]
        expected_shape = (chunk["frames"], HEIGHT, WIDTH)
        require(depth.shape == expected_shape, f"bad depth shape in {path}: {depth.shape}")
        require(semantic_id.shape == expected_shape, f"bad Semantic-ID shape in {path}: {semantic_id.shape}")
        require(semantic_id.dtype == np.dtype("uint8"), f"bad Semantic-ID dtype in {path}: {semantic_id.dtype}")
        require(int(semantic_id.max(initial=0)) <= 11, f"Semantic-ID outside [0,11] in {path}")
        if depth_format == METRIC_DEPTH:
            require(depth.dtype == np.dtype("float32"), f"bad metric depth dtype in {path}: {depth.dtype}")
            require(bool(np.all(np.isfinite(depth))) and not bool(np.any(depth < 0)), f"invalid metric depth in {path}")
            metric = np.ascontiguousarray(depth, dtype="<f4")
        else:
            require(depth.dtype == np.dtype("uint16"), f"bad fixed-log depth dtype in {path}: {depth.dtype}")
            metric = fixed_log_codes_to_metric(depth)
            require(np.array_equal(metric_to_fixed_log_codes(metric), depth), f"fixed-log round-trip failed in {path}")
        start = chunk["start_frame"]
        for offset in range(chunk["frames"]):
            yield start + offset, metric[offset], semantic_id[offset]


def materialize_example(
    source_root: Path,
    output_root: Path,
    example_id: int,
    *,
    require_config: bool,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    example_root = source_root / str(example_id)
    metadata_path = example_root / "metadata.json"
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    validate_metadata(example_root, metadata, example_id)
    destination = output_root / "examples" / f"example_{example_id:02d}"
    config_path = destination / "config.json"
    if require_config:
        ensure_release_config(config_path, example_root, metadata, example_id)
    _copy_exact(example_root / metadata["first_frame"], destination / "first_frame.png")
    written = 0
    for ordinal, depth, semantic_id in _iter_chunks(example_root, metadata):
        if frame_limit is not None and ordinal >= frame_limit:
            break
        condition_root = destination / "conditions"
        _atomic_bytes(condition_root / f"{ordinal:06d}.depth.f32", np.ascontiguousarray(depth, dtype="<f4").tobytes(order="C"))
        _atomic_semantic_png(condition_root / f"{ordinal:06d}.semantic_id.png", semantic_id)
        written += 1
    expected = min(metadata["frames"], frame_limit) if frame_limit is not None else metadata["frames"]
    require(written == expected, f"example {example_id}: materialized {written}, expected {expected}")
    return {
        "id": example_id,
        "depth_format": metadata["depth_format"],
        "frames": metadata["frames"],
        "materialized_frames": written,
        "windows": len(metadata["prompts"]),
    }


def parse_ids(values: list[int] | None) -> list[int]:
    ids = list(range(1, 41)) if not values else sorted(set(values))
    require(ids and all(1 <= value <= 40 for value in ids), "example ids must be in [1,40]")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path, help="downloaded code-world-model-inference-examples-40 directory")
    parser.add_argument("--output", required=True, type=Path, help="release root containing models/, examples/, and outputs/")
    parser.add_argument("--example", action="append", type=int, dest="examples", help="materialize one public example id; repeat as needed")
    parser.add_argument("--without-config-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frame-limit", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    manifest = _read_json(source / "manifest.json")
    require(manifest.get("schema") == "code_world_model_inference_examples_v3", "unsupported dataset manifest")
    require(manifest.get("count") == 40 and manifest.get("expected_count") == 40, "dataset manifest does not contain 40 examples")
    results = []
    for example_id in parse_ids(args.examples):
        result = materialize_example(
            source,
            output,
            example_id,
            require_config=not args.without_config_check,
            frame_limit=args.frame_limit,
        )
        results.append(result)
        print(f"example_{example_id:02d}: {result['materialized_frames']} frames", flush=True)
    receipt = {
        "schema": "cwm_h3_materialized_examples_v1",
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "examples": results,
    }
    if args.frame_limit is None and len(results) == 40:
        path = output / "examples" / "materialization.json"
        _atomic_bytes(path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
