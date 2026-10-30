from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture
def config_factory(tmp_path: Path):
    """Create a complete fake filesystem contract without any model payloads."""

    def create(*, window_count: int = 1, **overrides: object) -> Path:
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        for name in ("dit.safetensors", "video_vae.safetensors", "audio_vae.safetensors", "lora.safetensors"):
            (models / name).touch()
        (models / "text_encoder").mkdir(exist_ok=True)
        (models / "processor").mkdir(exist_ok=True)

        (tmp_path / "anchor.png").touch()
        (tmp_path / "condition").mkdir(exist_ok=True)
        payload: dict[str, object] = {
            "models": {
                "dit": "models/dit.safetensors",
                "video_vae": "models/video_vae.safetensors",
                "audio_vae": "models/audio_vae.safetensors",
                "text_encoder": "models/text_encoder",
                "processor": "models/processor",
                "lora": "models/lora.safetensors",
            },
            "anchor_image": "anchor.png",
            "condition_root": "condition",
            "cache_dir": "cache",
            "output": "result.mp4",
            "windows": [
                {"prompt": f"Window {index}", "start_frame": index * 90}
                for index in range(window_count)
            ],
            "seed": 42,
            "steps": 20,
        }
        payload.update(overrides)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return create
