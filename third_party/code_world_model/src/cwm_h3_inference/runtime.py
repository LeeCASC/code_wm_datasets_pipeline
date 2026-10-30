"""Pure inference layout construction and strict LoRA loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .constants import AUDIO_LATENT_SHAPE, VIDEO_LATENT_SHAPE


@dataclass(frozen=True)
class RuntimeBatch:
    layout: Any
    text_hidden_states: Any
    text_token_tags: Any


def build_runtime_batch(anchor: Any, reference: Any, hidden: Any, tags: Any) -> RuntimeBatch:
    import torch
    from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout

    if anchor.ndim != 5 or tuple(anchor.shape[:3]) != (1, 24, 1):
        raise ValueError(f"anchor must be [1,24,1,H,W], got {tuple(anchor.shape)}")
    if reference.ndim != 5 or tuple(reference.shape[:3]) != (1, 24, 37):
        raise ValueError(f"reference must be [1,24,37,H,W], got {tuple(reference.shape)}")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 5120:
        raise ValueError(f"text hidden states must be [1,L,5120], got {tuple(hidden.shape)}")
    if tags.shape != hidden.shape[:2] or tags.dtype != torch.int64:
        raise ValueError("text token tags must be int64 [1,L]")
    layout = build_h3_layout(
        task="ref2va",
        text_length=hidden.shape[1],
        target_video=H3VideoGeometry(*VIDEO_LATENT_SHAPE[2:]),
        target_audio_frames=AUDIO_LATENT_SHAPE[-1],
        references=(
            H3ReferenceGeometry("image", video=H3VideoGeometry(*anchor.shape[2:])),
            H3ReferenceGeometry("video", video=H3VideoGeometry(*reference.shape[2:])),
        ),
    )
    return RuntimeBatch(layout=layout, text_hidden_states=hidden, text_token_tags=tags)


def strict_merge_lora(transformer: Any, checkpoint: Path, expected_modules: int) -> dict[str, int]:
    import torch
    from safetensors.torch import load_file
    from musubi_tuner.networks import lora_minimax_h3
    from musubi_tuner.utils.lora_utils import filter_lora_state_dict

    state = filter_lora_state_dict(load_file(str(checkpoint)), None, None)
    network = lora_minimax_h3.create_arch_network_from_weights(
        1.0, state, unet=transformer, for_inference=True
    )
    if network.text_encoder_loras:
        raise ValueError("LoRA unexpectedly targets the text encoder")
    if len(network.unet_loras) != expected_modules:
        raise ValueError(
            f"LoRA module count mismatch: expected {expected_modules}, got {len(network.unet_loras)}"
        )
    expected_keys: set[str] = set()
    names: set[str] = set()
    for module in network.unet_loras:
        name = str(module.lora_name)
        if name in names or module.split_dims is not None or re.search(r"[^A-Za-z0-9_.]", name):
            raise ValueError(f"invalid LoRA target: {name}")
        names.add(name)
        expected_keys.update(
            {f"{name}.alpha", f"{name}.lora_down.weight", f"{name}.lora_up.weight"}
        )
    missing = expected_keys - set(state)
    unexpected = set(state) - expected_keys
    if missing or unexpected:
        raise ValueError(f"LoRA key mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    network.merge_to(None, transformer, state, dtype=torch.bfloat16, device="cpu")
    result = {"loaded_modules": len(names), "missing_keys": 0, "unexpected_keys": 0}
    del network, state
    return result
