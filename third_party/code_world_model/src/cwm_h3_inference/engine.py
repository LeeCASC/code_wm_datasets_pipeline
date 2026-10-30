"""Single-GPU Ref2VA inference with a fixed first frame and Retake34 continuation."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from typing import Any
import uuid

from .cache import cache_paths, validate_cache
from .config import RunConfig
from .constants import (
    AUDIO_CONDITION_CLEAN,
    AUDIO_LATENT_SHAPE,
    AUDIO_PREFIX_LATENTS,
    AUDIO_SHIFT,
    FPS,
    OVERLAP_AUDIO_SAMPLES,
    OVERLAP_FRAMES,
    RECOVERY_SCHEMA,
    REFERENCE_LATENT_SHAPE,
    REFERENCE_RUNTIME_SHAPE,
    RESULT_SCHEMA,
    STRIDE_FRAMES,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_LATENT_SHAPE,
    VIDEO_PREFIX_LATENTS,
    VIDEO_SHIFT,
    VISUAL_CONDITION_CLEAN,
    WINDOW_AUDIO_SAMPLES,
    WINDOW_FRAMES,
)
from .io import atomic_json
from .runtime import build_runtime_batch, strict_merge_lora


def _require_one_gpu(torch: Any) -> Any:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("generation requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _unique_key(handle: Any, *, exact: str | None = None, prefix: str | None = None) -> str:
    keys = list(handle.keys())
    selected = [key for key in keys if key == exact] if exact else [key for key in keys if key.startswith(str(prefix))]
    if len(selected) != 1:
        raise ValueError(f"expected one tensor key, got {keys}")
    return selected[0]


class Ref2VAEngine:
    def __init__(self, config: RunConfig) -> None:
        import torch
        from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
        from musubi_tuner.minimax_h3.model import load_h3_transformer
        from musubi_tuner.minimax_h3.video_vae import load_video_vae
        from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

        self.config = config
        self.torch = torch
        self.device = _require_one_gpu(torch)
        transformer = load_h3_transformer(
            config.models.dit,
            device="cpu",
            dtype=torch.bfloat16,
            attn_mode=config.attention_mode,
            split_attn=False,
            disable_mmap=False,
        )
        self.lora_gate = strict_merge_lora(
            transformer, config.models.lora, config.expected_lora_modules
        )
        swap = BlockSwapConfig(device=self.device, supports_backward=False, use_pinned_memory=True)
        transformer.enable_block_swap(config.blocks_to_swap, swap)
        transformer.move_to_device_except_swap_blocks(self.device)
        transformer.prepare_block_swap_before_forward()
        transformer.switch_block_swap_for_inference()
        transformer.eval().requires_grad_(False)
        self.transformer = transformer
        self.video_vae = load_video_vae(
            config.models.video_vae, device=self.device, dtype=torch.float16, disable_mmap=False
        ).eval().requires_grad_(False)
        self.audio_vae = load_audio_vae(
            config.models.audio_vae, device=self.device, dtype=torch.float32, disable_mmap=False
        ).eval().requires_grad_(False)

    def _load_window_cache(self, index: int):
        import torch
        from safetensors import safe_open

        paths = cache_paths(self.config, index)
        with safe_open(str(paths["anchor"]), framework="pt", device="cpu") as handle:
            anchor = handle.get_tensor(_unique_key(handle, prefix="latents_ref_"))
        with safe_open(str(paths["reference"]), framework="pt", device="cpu") as handle:
            reference = handle.get_tensor(_unique_key(handle, prefix="latents_ref_"))
        with safe_open(str(paths["qwen"]), framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor(_unique_key(handle, exact="varlen_mmh3_hidden_states_bfloat16"))
            tags = handle.get_tensor(_unique_key(handle, exact="varlen_mmh3_token_tags_int64"))
        if tuple(reference.shape) != REFERENCE_LATENT_SHAPE or reference.dtype != torch.float32:
            raise ValueError(f"window {index} DUV reference cache shape/dtype changed")
        reference = torch.nn.functional.pad(reference.unsqueeze(0), (0, 1, 0, 0), value=0.0).contiguous()
        if tuple(reference.shape) != REFERENCE_RUNTIME_SHAPE:
            raise ValueError(f"window {index} padded DUV reference shape changed")
        if anchor.ndim != 4 or tuple(anchor.shape[:2]) != (24, 1) or anchor.dtype != torch.float32:
            raise ValueError("anchor cache shape/dtype changed")
        if hidden.ndim != 2 or hidden.shape[1] != 5120 or hidden.dtype != torch.bfloat16:
            raise ValueError(f"window {index} Qwen hidden states changed")
        if tags.shape != (hidden.shape[0],) or tags.dtype != torch.int64:
            raise ValueError(f"window {index} Qwen token tags changed")
        if not all(torch.isfinite(value).all().item() for value in (anchor, reference, hidden)):
            raise ValueError(f"window {index} cache contains non-finite values")
        return anchor.unsqueeze(0), reference, hidden.unsqueeze(0), tags.unsqueeze(0)

    def _load_input_video(self):
        import torch
        from safetensors import safe_open

        path = cache_paths(self.config, 0)["input_video"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            latent = handle.get_tensor(_unique_key(handle, exact="input_video_latents"))
        if tuple(latent.shape) != VIDEO_LATENT_SHAPE[1:] or latent.dtype != torch.float32:
            raise ValueError("fixed-first-frame latent shape/dtype changed")
        if torch.count_nonzero(latent[:, 1:]).item() != 0:
            raise ValueError("fixed-first-frame latent must contain data only at temporal index 0")
        latent = latent.unsqueeze(0).to(self.device)
        mask = torch.ones_like(latent, dtype=torch.bool)
        mask[:, :, 0] = False
        return latent, mask

    def _recovery_path(self, index: int) -> Path:
        return self.config.cache_dir / "recovery" / f"window_{index:03d}.safetensors"

    def _load_recovery(self, index: int):
        import torch
        from safetensors import safe_open

        path = self._recovery_path(index)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("schema") != RECOVERY_SCHEMA or metadata.get("window_index") != str(index):
                raise ValueError(f"recovery identity changed: {path}")
            video = handle.get_tensor("video")
            audio = handle.get_tensor("audio")
        if tuple(video.shape) != VIDEO_LATENT_SHAPE[1:] or tuple(audio.shape) != AUDIO_LATENT_SHAPE[1:]:
            raise ValueError(f"recovery shape changed: {path}")
        if video.dtype != torch.float32 or audio.dtype != torch.float32:
            raise ValueError(f"recovery dtype changed: {path}")
        if not torch.isfinite(video).all().item() or not torch.isfinite(audio).all().item():
            raise ValueError(f"recovery contains non-finite values: {path}")
        return video, audio

    def _retake_inputs(self, previous: tuple[Any, Any]):
        torch = self.torch
        previous_video, previous_audio = previous
        torch.cuda.empty_cache()
        with torch.inference_mode():
            decoded_video = self.video_vae.decode(
                previous_video.unsqueeze(0).to(self.device, torch.float16)
            )[:, :, :WINDOW_FRAMES]
            if tuple(decoded_video.shape) != (1, 3, WINDOW_FRAMES, TARGET_HEIGHT, TARGET_WIDTH):
                raise RuntimeError("Retake34 decoded video shape changed")
            tail = decoded_video[:, :, -OVERLAP_FRAMES:]
            padded = torch.cat((tail, tail[:, :, -1:].repeat(1, 1, STRIDE_FRAMES, 1, 1)), dim=2)
            clean_video = self.video_vae.encode(padded.to(torch.float16)).float()
            decoded_audio = self.audio_vae.decode(
                previous_audio.unsqueeze(0).to(self.device, torch.float32)
            )[:, :, :WINDOW_AUDIO_SAMPLES]
            if tuple(decoded_audio.shape) != (1, 2, WINDOW_AUDIO_SAMPLES):
                raise RuntimeError("Retake34 decoded audio shape changed")
            prefix = decoded_audio[:, :, -OVERLAP_AUDIO_SAMPLES:]
            prefix_audio = self.audio_vae.encode(prefix.to(torch.float32)).float()
            if prefix_audio.shape[-1] != AUDIO_PREFIX_LATENTS:
                raise RuntimeError("Retake34 audio prefix shape changed")
            clean_audio = torch.cat(
                (prefix_audio, prefix_audio[..., -1:].repeat(1, 1, 1, AUDIO_LATENT_SHAPE[-1] - AUDIO_PREFIX_LATENTS)),
                dim=-1,
            )
        if tuple(clean_video.shape) != VIDEO_LATENT_SHAPE or tuple(clean_audio.shape) != AUDIO_LATENT_SHAPE:
            raise RuntimeError("Retake34 clean latent shape changed")
        video_mask = torch.ones_like(clean_video, dtype=torch.bool)
        video_mask[:, :, :VIDEO_PREFIX_LATENTS] = False
        audio_mask = torch.ones_like(clean_audio, dtype=torch.bool)
        audio_mask[..., :AUDIO_PREFIX_LATENTS] = False
        return clean_video, video_mask, clean_audio, audio_mask

    def sample_window(self, index: int) -> Path:
        import torch
        from musubi_tuner.minimax_h3.sampling import initialize_target_latents, sample_joint_av
        from safetensors.torch import save_file

        output = self._recovery_path(index)
        if output.exists():
            self._load_recovery(index)
            return output
        anchor, reference, hidden, tags = self._load_window_cache(index)
        runtime = build_runtime_batch(anchor, reference, hidden, tags)
        window_seed = self.config.seed + index
        initial_video, initial_audio = initialize_target_latents(
            video_shape=VIDEO_LATENT_SHAPE,
            audio_shape=AUDIO_LATENT_SHAPE,
            seed=window_seed,
            device=self.device,
            video_dtype=torch.float32,
            audio_dtype=torch.float32,
        )
        ref = reference.to(self.device, torch.float32)
        ref_generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 1000 + index)
        ref_noise = torch.randn(tuple(ref.shape), generator=ref_generator, dtype=torch.float32).to(self.device)
        augmented_ref = VISUAL_CONDITION_CLEAN * ref + (1.0 - VISUAL_CONDITION_CLEAN) * ref_noise
        anchor_ref = anchor.to(self.device, torch.float32)
        anchor_generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 3000 + index)
        anchor_noise = torch.randn(tuple(anchor_ref.shape), generator=anchor_generator, dtype=torch.float32).to(self.device)
        augmented_anchor = VISUAL_CONDITION_CLEAN * anchor_ref + (1.0 - VISUAL_CONDITION_CLEAN) * anchor_noise
        retake = self._load_input_video() + (None, None) if index == 0 else self._retake_inputs(self._load_recovery(index - 1))
        with torch.inference_mode():
            sampled = sample_joint_av(
                self.transformer,
                layout=runtime.layout,
                text_hidden_states=runtime.text_hidden_states.to(self.device, torch.bfloat16),
                text_token_tags=runtime.text_token_tags.to(self.device),
                initial_video=initial_video,
                initial_audio=initial_audio,
                steps=self.config.steps,
                video_shift=VIDEO_SHIFT,
                audio_shift=AUDIO_SHIFT,
                visual_condition_latents=(augmented_anchor, augmented_ref),
                audio_condition_latents=(),
                visual_condition_clean=VISUAL_CONDITION_CLEAN,
                audio_condition_clean=AUDIO_CONDITION_CLEAN,
                input_video_latents=retake[0],
                video_denoise_mask=retake[1],
                input_audio_latents=retake[2],
                audio_denoise_mask=retake[3],
            )
        video = sampled.video[0].float().cpu().contiguous()
        audio = sampled.audio[0].float().cpu().contiguous()
        if not torch.isfinite(video).all().item() or not torch.isfinite(audio).all().item():
            raise RuntimeError(f"window {index} produced non-finite latents")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            save_file(
                {"video": video, "audio": audio},
                str(temporary),
                metadata={
                    "schema": RECOVERY_SCHEMA,
                    "window_index": str(index),
                    "seed": str(window_seed),
                    "steps": str(self.config.steps),
                },
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
            del sampled, video, audio, anchor, reference, hidden, tags, runtime
            del initial_video, initial_audio, ref, ref_noise, augmented_ref
            del anchor_ref, anchor_noise, augmented_anchor, retake
            gc.collect()
            torch.cuda.empty_cache()
        return output

    def write_video(self) -> dict[str, Any]:
        import av
        import torch

        output = self.config.output
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4")
        frame_count = 0
        try:
            with av.open(str(temporary), mode="w") as container:
                stream = container.add_stream("libx264", rate=FPS)
                stream.width = TARGET_WIDTH
                stream.height = TARGET_HEIGHT
                stream.pix_fmt = "yuv420p"
                for index in range(len(self.config.windows)):
                    video, _audio = self._load_recovery(index)
                    torch.cuda.empty_cache()
                    with torch.inference_mode():
                        decoded = self.video_vae.decode(
                            video.unsqueeze(0).to(self.device, torch.float16)
                        )[0, :, :WINDOW_FRAMES].cpu()
                    pixels = (
                        ((decoded.float().clamp(-1, 1) + 1.0) * 127.5)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 3, 0)
                        .contiguous()
                    )
                    if index:
                        pixels = pixels[OVERLAP_FRAMES:]
                    for value in pixels:
                        frame = av.VideoFrame.from_ndarray(value.numpy(), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        frame_count += 1
                    del video, decoded, pixels
                    gc.collect()
                    torch.cuda.empty_cache()
                for packet in stream.encode():
                    container.mux(packet)
            if frame_count != self.config.output_frames:
                raise RuntimeError(f"stitched frame count mismatch: {frame_count}")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = {
            "schema": RESULT_SCHEMA,
            "solver": "explicit_euler",
            "steps": self.config.steps,
            "seed": self.config.seed,
            "window_frames": WINDOW_FRAMES,
            "overlap_frames": OVERLAP_FRAMES,
            "output_frames": frame_count,
            "fps": FPS,
            "resolution": [TARGET_WIDTH, TARGET_HEIGHT],
            "audio": False,
            "output": str(output),
            "bytes": output.stat().st_size,
            "lora_gate": self.lora_gate,
        }
        atomic_json(output.with_suffix(output.suffix + ".json"), receipt)
        return receipt


def generate(config: RunConfig) -> dict[str, Any]:
    validate_cache(config)
    started = time.monotonic()
    engine = Ref2VAEngine(config)
    for index in range(len(config.windows)):
        engine.sample_window(index)
    result = engine.write_video()
    result["elapsed_seconds"] = time.monotonic() - started
    return result
