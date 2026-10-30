"""Prepare the DUV, RGB-anchor, fixed-first-frame, and Qwen caches."""

from __future__ import annotations

import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .constants import (
    ANCHOR_SHORT_EDGE,
    CACHE_SCHEMA,
    CANVAS_MULTIPLE,
    CONDITION_HEIGHT,
    CONDITION_WIDTH,
    FPS,
    QWEN_OFFSETS,
    REFERENCE_LATENT_SHAPE,
    SINGLE_FRAME_LATENT_SHAPE,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_LATENT_SHAPE,
)
from .duv import load_duv_window
from .io import atomic_json, atomic_safetensors, load_json
from .presentation import _system_prompt, build_ref2va_presentation


def official_short_edge_size(width: int, height: int) -> tuple[int, int]:
    if min(width, height) <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError("anchor aspect ratio must be between 1:4 and 4:1")
    scale = ANCHOR_SHORT_EDGE / min(width, height)

    def round_axis(value: float) -> int:
        return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    return round_axis(width * scale), round_axis(height * scale)


def _load_anchor(path: Path, *, size: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        image = image.convert("RGB")
        target = official_short_edge_size(*image.size) if size is None else size
        pixels = np.asarray(image.resize(target, Image.Resampling.LANCZOS), dtype=np.uint8).copy()
    return pixels


def cache_paths(config: RunConfig, window_index: int) -> dict[str, Path]:
    root = config.cache_dir
    window = root / f"window_{window_index:03d}"
    return {
        "anchor": root / "anchor_reference.safetensors",
        "input_video": root / "input_video.safetensors",
        "reference": window / "duv_reference.safetensors",
        "qwen": window / "qwen.safetensors",
        "manifest": root / "manifest.json",
    }


def _cache_contract(config: RunConfig) -> dict[str, Any]:
    return {
        "video_vae": str(config.models.video_vae),
        "text_encoder": str(config.models.text_encoder),
        "processor": str(config.models.processor),
        "anchor_image": str(config.anchor_image),
        "anchor_bytes": config.anchor_image.stat().st_size,
        "condition_root": str(config.condition_root),
        "system_prompt_w0": _system_prompt(0),
        "system_prompt_wn": _system_prompt(1),
        "windows": [
            {"prompt": window.prompt, "start_frame": window.start_frame}
            for window in config.windows
        ],
    }


def _contract_id(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _begin_manifest(config: RunConfig) -> dict[str, Any]:
    path = cache_paths(config, 0)["manifest"]
    contract = _cache_contract(config)
    contract_id = _contract_id(contract)
    if path.exists():
        existing = load_json(path)
        if existing.get("schema") != CACHE_SCHEMA or existing.get("contract_id") != contract_id:
            raise ValueError(f"cache directory belongs to another run: {config.cache_dir}")
        return existing
    if config.cache_dir.exists() and any(config.cache_dir.iterdir()):
        raise ValueError(f"non-empty cache directory has no manifest: {config.cache_dir}")
    manifest = {
        "schema": CACHE_SCHEMA,
        "status": "preparing",
        "contract_id": contract_id,
        "contract": contract,
        "window_count": len(config.windows),
        "output_frames": config.output_frames,
        "files": [],
    }
    atomic_json(path, manifest)
    return manifest


def _require_one_gpu(torch: Any) -> Any:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("cache preparation requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _prepare_visual_caches(config: RunConfig) -> None:
    import torch
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels

    device = _require_one_gpu(torch)
    vae = load_video_vae(config.models.video_vae, device=device, dtype=torch.float16, disable_mmap=False)
    vae.eval().requires_grad_(False)
    common = {"schema": CACHE_SCHEMA}
    first_paths = cache_paths(config, 0)

    if not first_paths["anchor"].exists():
        anchor_pixels = _load_anchor(config.anchor_image)
        frames = torch.from_numpy(anchor_pixels).unsqueeze(0)
        with torch.inference_mode():
            latent = encode_video_condition(vae, _prepare_pixels(frames).to(device, torch.float16))[0].cpu()
        if latent.ndim != 4 or tuple(latent.shape[:2]) != (24, 1):
            raise RuntimeError(f"unexpected anchor latent shape: {tuple(latent.shape)}")
        atomic_safetensors(
            first_paths["anchor"],
            {"latents_ref_000_image": latent.float().contiguous()},
            {**common, "kind": "anchor", "source": config.anchor_image.name},
        )
        del anchor_pixels, frames, latent

    if not first_paths["input_video"].exists():
        pixels = _load_anchor(config.anchor_image, size=(TARGET_WIDTH, TARGET_HEIGHT))
        frames = torch.from_numpy(pixels).unsqueeze(0)
        prepared = _prepare_pixels(frames).to(device, torch.float16)
        with torch.inference_mode():
            single = vae.encode(prepared).float().cpu()
        if tuple(single.shape) == (1, *SINGLE_FRAME_LATENT_SHAPE):
            single = single[0]
        if tuple(single.shape) != SINGLE_FRAME_LATENT_SHAPE:
            raise RuntimeError(f"unexpected first-frame latent shape: {tuple(single.shape)}")
        full = torch.zeros(VIDEO_LATENT_SHAPE[1:], dtype=torch.float32)
        full[:, 0:1] = single
        atomic_safetensors(
            first_paths["input_video"],
            {"input_video_latents": full.contiguous()},
            {
                **common,
                "kind": "fixed_first_frame",
                "fixed_video_latent_indices": "0",
                "denoise_mask_semantics": "0_fixed_1_regenerate",
            },
        )
        del pixels, frames, prepared, single, full

    for index, window in enumerate(config.windows):
        path = cache_paths(config, index)["reference"]
        if path.exists():
            continue
        visual = load_duv_window(config.condition_root, window.start_frame, for_qwen=False)
        pixels = _prepare_pixels(torch.from_numpy(visual.copy())).to(device, torch.float16)
        with torch.inference_mode():
            latent = encode_video_condition(vae, pixels)[0].cpu()
        if tuple(latent.shape) != REFERENCE_LATENT_SHAPE:
            raise RuntimeError(f"unexpected DUV reference shape for window {index}: {tuple(latent.shape)}")
        atomic_safetensors(
            path,
            {"latents_ref_001_video": latent.float().contiguous()},
            {**common, "kind": "duv_reference", "window_index": str(index)},
        )
        del visual, pixels, latent
        gc.collect()
        torch.cuda.empty_cache()

    del vae
    gc.collect()
    torch.cuda.empty_cache()


def _prepare_qwen_caches(config: RunConfig) -> None:
    import numpy as np
    import torch
    from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
    from musubi_tuner.minimax_h3.text_encoder import (
        H3TextVisual,
        build_presentation,
        encode_h3_presentation,
        load_h3_processor,
        load_h3_text_encoder,
    )

    pending = [index for index in range(len(config.windows)) if not cache_paths(config, index)["qwen"].exists()]
    if not pending:
        return
    device = _require_one_gpu(torch)
    processor = load_h3_processor(str(config.models.processor))
    model = load_h3_text_encoder(
        str(config.models.text_encoder),
        processor_path=str(config.models.processor),
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False)
    anchor_np = _load_anchor(config.anchor_image)
    anchor = torch.from_numpy(np.ascontiguousarray(anchor_np)).unsqueeze(0)
    for index in pending:
        window = config.windows[index]
        duv_np = load_duv_window(config.condition_root, window.start_frame, for_qwen=True)
        duv = torch.from_numpy(np.ascontiguousarray(duv_np))
        image_key = config.anchor_image
        video_key = config.condition_root / f"{window.start_frame:06d}.semantic_id.png"
        placeholder = config.condition_root / ".h3-presentation-placeholder"
        record = H3Record(
            video_path=placeholder,
            caption=window.prompt,
            target_audio=H3AudioSource(path=placeholder, embedded=False),
            references=(
                H3Reference(type="image", path=image_key),
                H3Reference(type="video", path=video_key, duration_seconds=124 / FPS),
            ),
            jsonl_line=0,
        )
        plain = build_presentation(
            record,
            "ref2va",
            {
                image_key: H3TextVisual(anchor),
                video_key: H3TextVisual(duv, tuple(offset / FPS for offset in QWEN_OFFSETS)),
            },
        )
        presentation = build_ref2va_presentation(processor, plain, index, window.prompt)
        with torch.inference_mode():
            hidden, tags = encode_h3_presentation(processor, model, presentation)
        hidden = hidden.to(device="cpu", dtype=torch.bfloat16).contiguous()
        tags = tags.to(device="cpu", dtype=torch.int64).contiguous()
        if hidden.ndim != 2 or hidden.shape[1] != 5120 or tags.shape != (hidden.shape[0],):
            raise RuntimeError(f"unexpected Qwen output shape for window {index}")
        atomic_safetensors(
            cache_paths(config, index)["qwen"],
            {"varlen_mmh3_hidden_states_bfloat16": hidden, "varlen_mmh3_token_tags_int64": tags},
            {
                "schema": CACHE_SCHEMA,
                "kind": "qwen",
                "window_index": str(index),
                "system_prompt_role": "w0" if index == 0 else "wn",
                "presentation_order": "picture1_anchor_then_video1_duv",
            },
        )
        del duv_np, duv, plain, presentation, hidden, tags
        gc.collect()
        torch.cuda.empty_cache()

    del anchor_np, anchor, processor, model
    gc.collect()
    torch.cuda.empty_cache()


def prepare_caches(config: RunConfig) -> dict[str, Any]:
    config.validate_files()
    initial = _begin_manifest(config)
    if initial.get("status") == "complete":
        return validate_cache(config)
    _prepare_visual_caches(config)
    _prepare_qwen_caches(config)
    files = []
    for index in range(len(config.windows)):
        paths = cache_paths(config, index)
        files.extend([paths["reference"], paths["qwen"]])
    files.extend([cache_paths(config, 0)["anchor"], cache_paths(config, 0)["input_video"]])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"cache preparation incomplete: {missing}")
    manifest = {
        "schema": CACHE_SCHEMA,
        "status": "complete",
        "contract_id": initial["contract_id"],
        "contract": initial["contract"],
        "window_count": len(config.windows),
        "output_frames": config.output_frames,
        "files": [str(path) for path in files],
    }
    atomic_json(cache_paths(config, 0)["manifest"], manifest)
    return manifest


def validate_cache(config: RunConfig) -> dict[str, Any]:
    manifest = load_json(cache_paths(config, 0)["manifest"])
    expected_id = _contract_id(_cache_contract(config))
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("contract_id") != expected_id
    ):
        raise ValueError("cache manifest is incompatible with this release")
    if manifest.get("window_count") != len(config.windows):
        raise ValueError("cache window count does not match config")
    expected_files: list[Path] = []
    for index in range(len(config.windows)):
        paths = cache_paths(config, index)
        expected_files.extend([paths["reference"], paths["qwen"]])
    expected_files.extend([cache_paths(config, 0)["anchor"], cache_paths(config, 0)["input_video"]])
    files = manifest.get("files")
    if files != [str(path) for path in expected_files]:
        raise ValueError("cache manifest file set does not match config")
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise ValueError(f"cache files are missing: {missing}")
    return manifest
