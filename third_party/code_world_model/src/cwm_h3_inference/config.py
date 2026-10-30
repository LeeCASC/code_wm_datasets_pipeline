"""Structured, path-parameterized configuration for one inference run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_STEPS, EXPECTED_LORA_MODULES, STRIDE_FRAMES, WINDOW_FRAMES


ALLOWED_ROOT_KEYS = {
    "models",
    "anchor_image",
    "condition_root",
    "cache_dir",
    "output",
    "windows",
    "seed",
    "steps",
    "attention_mode",
    "blocks_to_swap",
    "expected_lora_modules",
}
ALLOWED_MODEL_KEYS = {"dit", "video_vae", "audio_vae", "text_encoder", "processor", "lora"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(value: Any, label: str, *, base: Path) -> Path:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _canonical_caption(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    # The released Qwen caches were encoded from UTF-8 caption bytes whose
    # internal line endings are CRLF. JSON editors and Path.read_text() commonly
    # normalize those bytes to LF, which changes Qwen token IDs. Canonicalize the
    # user caption here so both representations consume the released text.
    normalized = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n")


@dataclass(frozen=True)
class ModelPaths:
    dit: Path
    video_vae: Path
    audio_vae: Path
    text_encoder: Path
    processor: Path
    lora: Path


@dataclass(frozen=True)
class Window:
    prompt: str
    start_frame: int


@dataclass(frozen=True)
class RunConfig:
    source: Path
    models: ModelPaths
    anchor_image: Path
    condition_root: Path
    cache_dir: Path
    output: Path
    windows: tuple[Window, ...]
    seed: int
    steps: int
    attention_mode: str
    blocks_to_swap: int
    expected_lora_modules: int

    @property
    def output_frames(self) -> int:
        from .constants import STRIDE_FRAMES

        return WINDOW_FRAMES + (len(self.windows) - 1) * STRIDE_FRAMES

    def validate_files(self) -> None:
        for label, path in (
            ("dit", self.models.dit),
            ("video_vae", self.models.video_vae),
            ("audio_vae", self.models.audio_vae),
            ("text_encoder", self.models.text_encoder),
            ("processor", self.models.processor),
            ("lora", self.models.lora),
            ("anchor_image", self.anchor_image),
            ("condition_root", self.condition_root),
        ):
            require(path.exists(), f"{label} does not exist: {path}")
        from .presentation import _system_prompt

        _system_prompt(0)
        _system_prompt(1)


def load_config(path: str | Path, *, check_files: bool = False) -> RunConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    require(isinstance(raw, dict), "config root must be an object")
    require(not (set(raw) - ALLOWED_ROOT_KEYS), f"unknown config keys: {sorted(set(raw) - ALLOWED_ROOT_KEYS)}")
    base = source.parent

    model_raw = raw.get("models")
    require(isinstance(model_raw, dict), "models must be an object")
    require(
        set(model_raw) == ALLOWED_MODEL_KEYS,
        f"models must contain exactly: {sorted(ALLOWED_MODEL_KEYS)}",
    )
    models = ModelPaths(
        dit=_path(model_raw.get("dit"), "models.dit", base=base),
        video_vae=_path(model_raw.get("video_vae"), "models.video_vae", base=base),
        audio_vae=_path(model_raw.get("audio_vae"), "models.audio_vae", base=base),
        text_encoder=_path(model_raw.get("text_encoder"), "models.text_encoder", base=base),
        processor=_path(model_raw.get("processor"), "models.processor", base=base),
        lora=_path(model_raw.get("lora"), "models.lora", base=base),
    )

    window_raw = raw.get("windows")
    require(isinstance(window_raw, list) and bool(window_raw), "windows must be a non-empty list")
    windows: list[Window] = []
    for index, item in enumerate(window_raw):
        require(isinstance(item, dict), f"windows[{index}] must be an object")
        prompt = item.get("prompt")
        start = item.get("start_frame")
        prompt = _canonical_caption(prompt, f"windows[{index}].prompt")
        require(type(start) is int and start >= 0, f"windows[{index}].start_frame must be non-negative")
        require(set(item) == {"prompt", "start_frame"}, f"windows[{index}] has unknown or missing keys")
        if windows:
            require(
                start == windows[-1].start_frame + STRIDE_FRAMES,
                f"windows[{index}].start_frame must advance by {STRIDE_FRAMES}",
            )
        windows.append(Window(prompt=prompt, start_frame=start))

    seed = raw.get("seed", 0)
    steps = raw.get("steps", DEFAULT_STEPS)
    attention_mode = raw.get("attention_mode", "flash3")
    blocks_to_swap = raw.get("blocks_to_swap", 48)
    expected_lora_modules = raw.get("expected_lora_modules", EXPECTED_LORA_MODULES)
    require(type(seed) is int and 0 <= seed < 2**63, "seed must be an integer in [0, 2**63)")
    require(type(steps) is int and steps == DEFAULT_STEPS, "this release fixes explicit Euler inference to 20 steps")
    require(attention_mode == "flash3", "this release fixes attention_mode to flash3")
    require(type(blocks_to_swap) is int and 0 <= blocks_to_swap <= 48, "blocks_to_swap must be in [0,48]")
    require(
        type(expected_lora_modules) is int and expected_lora_modules == EXPECTED_LORA_MODULES,
        f"expected_lora_modules must be {EXPECTED_LORA_MODULES}",
    )

    config = RunConfig(
        source=source,
        models=models,
        anchor_image=_path(raw.get("anchor_image"), "anchor_image", base=base),
        condition_root=_path(raw.get("condition_root"), "condition_root", base=base),
        cache_dir=_path(raw.get("cache_dir", "cache"), "cache_dir", base=base),
        output=_path(raw.get("output", "output.mp4"), "output", base=base),
        windows=tuple(windows),
        seed=seed,
        steps=steps,
        attention_mode=attention_mode,
        blocks_to_swap=blocks_to_swap,
        expected_lora_modules=expected_lora_modules,
    )
    if check_files:
        config.validate_files()
    return config
