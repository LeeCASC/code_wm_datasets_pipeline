from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from cwm_h3_inference.cache import _begin_manifest, cache_paths, official_short_edge_size
from cwm_h3_inference.cli import build_parser, main
from cwm_h3_inference.config import load_config
from cwm_h3_inference.constants import CACHE_SCHEMA, OVERLAP_FRAMES, WINDOW_FRAMES
from cwm_h3_inference.presentation import _system_prompt, build_ref2va_presentation


def test_config_accepts_fixed_inference_settings(config_factory) -> None:
    config = load_config(config_factory(), check_files=True)

    assert config.steps == 20
    assert config.source.is_absolute()
    assert config.anchor_image.is_absolute()
    assert _system_prompt(0)
    assert _system_prompt(1)
    assert _system_prompt(0) != _system_prompt(1)


def test_cache_schema_is_cwm_and_rejects_incompatible_manifest(config_factory) -> None:
    config = load_config(config_factory())
    manifest = _begin_manifest(config)

    assert CACHE_SCHEMA == "cwm_h3_inference_cache_v3"
    assert manifest["schema"] == CACHE_SCHEMA

    path = cache_paths(config, 0)["manifest"]
    manifest["schema"] = "incompatible_cache_schema"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cache directory belongs to another run"):
        _begin_manifest(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "unsupported"}, "unknown config keys"),
        ({"steps": 19}, "explicit Euler inference to 20 steps"),
        ({"steps": 40}, "explicit Euler inference to 20 steps"),
        ({"attention_mode": "sdpa"}, "attention_mode to flash3"),
        ({"expected_lora_modules": 199}, "expected_lora_modules must be 200"),
    ],
)
def test_config_rejects_unknown_keys_and_changed_fixed_settings(config_factory, override, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(config_factory(**override))


@pytest.mark.parametrize(
    ("window_count", "expected"),
    [
        (1, 124),
        (3, 304),
        (8, 754),
    ],
)
def test_output_frames_applies_retake34_overlap(config_factory, window_count, expected) -> None:
    config = load_config(config_factory(window_count=window_count))

    assert WINDOW_FRAMES == 124
    assert OVERLAP_FRAMES == 34
    assert config.output_frames == expected


def test_config_rejects_non_contiguous_condition_windows(config_factory) -> None:
    path = config_factory(window_count=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["windows"][1]["start_frame"] = 91
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must advance by 90"):
        load_config(path)


@pytest.mark.parametrize("prompt", ["line one\n\nline two", "line one\r\n\r\nline two"])
def test_config_canonicalizes_caption_line_endings(config_factory, prompt) -> None:
    path = config_factory()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["windows"][0]["prompt"] = prompt
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(path)

    assert config.windows[0].prompt == "line one\r\n\r\nline two"


def test_config_rejects_unknown_control_override(config_factory) -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(config_factory(custom_control="custom.txt"))


def test_presentation_has_no_prompt_injection_parameter() -> None:
    assert tuple(inspect.signature(build_ref2va_presentation).parameters) == (
        "processor",
        "base",
        "window_index",
        "caption",
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((1536, 864), (3648, 2048)),
        ((864, 1536), (2048, 3648)),
        ((2048, 2048), (2048, 2048)),
    ],
)
def test_official_short_edge_size(source, expected) -> None:
    result = official_short_edge_size(*source)

    assert result == expected
    assert min(result) == 2048
    assert all(axis % 32 == 0 for axis in result)


@pytest.mark.parametrize("source", [(0, 1080), (-1, 1080), (5000, 999), (999, 5000)])
def test_official_short_edge_size_rejects_invalid_geometry(source) -> None:
    with pytest.raises(ValueError, match="aspect ratio"):
        official_short_edge_size(*source)


def test_cli_validate_uses_only_fake_local_files(config_factory, capsys) -> None:
    result = main(["validate", "--config", str(config_factory(window_count=3))])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "pass",
        "window_count": 3,
        "output_frames": 304,
    }


def test_cli_exposes_only_the_three_commands() -> None:
    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == "command")

    assert set(action.choices) == {"validate", "prepare", "generate"}


def test_examples_use_the_supported_contract() -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in examples.glob("*.json")]

    assert payloads
    assert all("mode" not in payload for payload in payloads)
    assert {payload["steps"] for payload in payloads} == {20}
