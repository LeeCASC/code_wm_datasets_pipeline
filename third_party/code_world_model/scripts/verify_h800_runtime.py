#!/usr/bin/env python3
"""Quick smoke test for the tested H800 runtime."""

from __future__ import annotations

import json


EXPECTED = {
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "cuda_runtime": "12.8",
}


def main() -> None:
    import av
    import torch
    import torchvision
    from flash_attn_interface import flash_attn_func

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("make exactly one CUDA GPU visible for this check")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    observed = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if observed != EXPECTED:
        raise RuntimeError(f"version mismatch: expected={EXPECTED} observed={observed}")
    if capability != (9, 0) or "H800" not in gpu:
        raise RuntimeError(f"tested runtime requires H800 SM90, got {gpu} {capability}")

    generator = torch.Generator(device="cpu").manual_seed(20260826)
    q, k, v = [
        torch.randn((1, 128, 4, 128), generator=generator, dtype=torch.bfloat16).cuda()
        for _ in range(3)
    ]
    with torch.inference_mode():
        output = flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()
    if output.shape != q.shape or not torch.isfinite(output).all().item():
        raise RuntimeError("FlashAttention-3 BF16 forward failed")

    av.codec.Codec("libx264", "w")
    print(
        json.dumps(
            {
                "status": "pass",
                **observed,
                "gpu": gpu,
                "capability": list(capability),
                "fa3_shape": list(output.shape),
                "libx264": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
