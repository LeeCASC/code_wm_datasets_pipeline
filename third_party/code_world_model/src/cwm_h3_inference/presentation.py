"""Ref2VA chat presentation with Picture 1 before Video 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


_PROMPT_ROOT = Path(__file__).with_name("prompts")
_PROMPT_PATHS = (
    _PROMPT_ROOT / "system_w0.txt",
    _PROMPT_ROOT / "system_wn.txt",
)
_PROMPT_SHA256 = (
    "d488897872a5b190ff8d56b6acc255b66a78f67ae862c12582fcffc8a5dd4ddc",
    "cd018def9793b4f73cd1c9da9d8ca9ca508dd94a260a19d915aaf9f80b97b6b2",
)


def _system_prompt(window_index: int) -> str:
    if type(window_index) is not int or window_index < 0:
        raise ValueError("window_index must be a non-negative integer")
    prompt_index = 0 if window_index == 0 else 1
    path = _PROMPT_PATHS[prompt_index]
    if not path.is_file():
        raise ValueError(f"packaged system prompt file is missing: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _PROMPT_SHA256[prompt_index]:
        raise ValueError(f"packaged system prompt does not match this release: {path.name}")
    prompt = payload.decode("utf-8").strip()
    if not prompt:
        raise ValueError(f"packaged system prompt is empty: {path.name}")
    return prompt


def build_ref2va_presentation(processor: Any, base: Any, window_index: int, caption: str):
    from musubi_tuner.minimax_h3.text_encoder import H3Presentation, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

    system_prompt = _system_prompt(window_index)
    if len(base.videos) != 1 or len(base.images) != 1:
        raise ValueError("Ref2VA requires exactly one image and one video reference")
    text = base.processor_text
    if text.count(IMAGE_PLACEHOLDER) != 1 or text.count(VIDEO_PLACEHOLDER) != 1:
        raise ValueError("Ref2VA presentation requires one image and one video placeholder")
    if text.count("<Picture 1>") != 1 or text.count("<Video 1>") != 1:
        raise ValueError("Ref2VA presentation labels changed")
    if text.index("<Picture 1>") >= text.index("<Video 1>"):
        raise ValueError("Picture 1 must precede Video 1")
    if text.count(caption) != 1:
        raise ValueError("caption must occur exactly once before chat formatting")
    formatted = processor.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if formatted.count(system_prompt) != 1 or formatted.count(caption) != 1:
        raise ValueError("system prompt and caption must each occur exactly once")
    if formatted.count("<|im_start|>system\n") != 1 or formatted.count("<|im_start|>user\n") != 1:
        raise ValueError("chat template must contain exactly one system block and one user block")
    if formatted.count("<|im_start|>assistant\n") != 0:
        raise ValueError("chat template must not add an assistant block")
    return H3Presentation(text=base.text, images=base.images, videos=base.videos, processor_text=formatted)
