# Actual pipeline contracts

## Aircraft

1. Python creates a 174-vertex / 292-triangle aircraft mesh, scripted body motion, and camera paths. The OBB control uses its body-coordinate bounds `[-9,-10,-1.8]` to `[9,10,3.4]` metres; it rotates with the body. It is not a minimum-volume OBB fit.
2. The CPU renderer exports camera-Z depth in metres as little-endian float32 `[192,336]` and semantic ID PNG `[192,336]`. RGB proxy previews are diagnostic only.
3. DUV encodes log depth and semantic IDs into three channels. Qwen receives an 8-bit representation at 11 sampled frame offsets; the video VAE receives all 124 frames with higher-precision normalized depth. The VAE cache shape is `[24,37,12,21]`, padded to width 22 for the runtime layout.
4. The supplied appearance anchor is 1672×941 in the included example. CWM's reference-image preprocessing resizes it to 3648×2048, producing `[24,1,128,228]`; a separate 1344×768 version supplies the fixed first latent frame. Upsampling does not create new source detail.
5. The H3-adapted Qwen3-VL-32B encodes prompts and visual references. Its condition feature width is 5120; sequence length varies. The daylight-terminal experiment had 8815 rows.
6. H3 Ref2VA DiT with the CWM LoRA merged into 200 target modules samples video and audio latents for 20 steps. RGB output video latent shape is `[24,37,48,84]`, omitting batch. This does not mean 37 output frames.
7. H3 video VAE decodes 124 RGB frames at 1344×768, 24 fps. The RGB deliverable is silent. Audio latent sampling is part of the native joint model, but the CWM export does not output an audio track.

This project uses the PyTorch FLASH SDPA backend on A100 rather than the upstream FlashAttention-3-only configuration. Numeric parity with the original backend has not been established.

## UE input format

```text
ue_capture/
  Observer07_OrbitRiseSlow/
    metadata.json
    frames.jsonl
    rgb/rgb.mp4
    rgb/poses.jsonl
  Observer08_OrbitFallSlow/
    ...
```

The implemented Observer demo expects 300 frames at 30 fps and the original schema with `time_seconds` and `camera_to_world_ue` in each `frames.jsonl` row, and `intrinsics.matrix_pixels` in metadata. Per-frame PNG paths in old metadata are not used. The supplied capture had RGB and poses, without depth or semantic maps.

- Select 240 source frames using `floor(frame_24fps * 30/24 + .5)`. Save the actual selected source frame's pose and time, not an interpolated pose pretending to occur at the selected frame time.
- Convert UE centimetres to metres and explicitly convert camera axes to OpenCV. Rescale intrinsics with pixel-center offsets when resizing.
- Segments use global frame ranges `[0:124]` and `[116:240]`. Each RGB reference is 672×384; generated output is 1344×768.
- A photographic anchor per segment is supplied externally. The existing four example anchors match the previously inspected Observer scenes only.
- Run native H3 Ref2VA with its adapted Qwen encoder, video/audio VAEs, RGB video reference, image reference and caption. No CWM LoRA is used for this baseline.
- Assemble `seg01[:120] + seg02[4:124]` to obtain 240 frames / 10 seconds. This is a hard cut, not a certified seamless sequence.

This schema's silent WAV is only a target-record placeholder. It is not supplied as an audio reference condition or labelled as GT.

## What trajectories mean

`flight_trajectory.json` and aircraft camera files describe the exact programmed proxy. UE camera mappings describe original selected RGB frames. Neither automatically certifies generated-pixel camera poses, newly invented objects, or cross-view consistency.

The completed UE RGB-reference experiments showed substantial camera/framing drift. A future CWM comparison requires frame-aligned metric depth and semantic maps re-exported from UE at the same source camera/times. It has not yet been performed.
