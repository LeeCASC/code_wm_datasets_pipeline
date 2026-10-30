# Inference guide

## Tested settings

This release supports the following tested configuration:

- MiniMax H3 BF16 Ref2VA with a compatible CWM LoRA;
- one RGB `<Picture 1>` anchor followed by one Depth + Semantic-ID `<Video 1>`
  reference;
- 1344x768 output at 24 FPS;
- 124 frames per window, Retake34 overlap, and 90 new frames per continuation;
- explicit Euler sampling with exactly 20 steps;
- FlashAttention-3 on one visible NVIDIA H800 GPU; and
- silent H.264 MP4 output.

The model still samples audio latents as part of its packed layout and
continuation state. An Audio VAE is therefore required even though the final
MP4 has no audio stream.

## Input format

For released inference, start from one of the downloaded multi-window
configurations at `release/examples/example_XX/config.json`. The installation
script obtains the inputs from the
[40 compact Hugging Face examples](https://huggingface.co/datasets/NTU-yiwen/code-world-model-inference-examples-40)
and deterministically expands their NPZ chunks into the per-frame layout below.
The bundled [multi-window configuration](examples/config.multiwindow.example.json)
is a template for custom inputs. The bundled
[single-window configuration](examples/config.example.json) is retained only
as a minimal smoke-test template. Relative paths are resolved from the
configuration file's directory.

The `condition_root` is a flat directory containing one pair per source-frame
ordinal:

```text
000000.depth.f32
000000.semantic_id.png
000001.depth.f32
000001.semantic_id.png
...
```

Each depth file is a headerless, C-order, little-endian float32 array of shape
`192x336` in metres and is exactly 258,048 bytes. Values must be finite and
non-negative. Values at or below 0.001 metres are invalid; input producers
should store zero for invalid depth.

Each Semantic-ID file is an 8-bit grayscale PNG at 336x192 with values in
`[0, 11]`:

```text
0 void_unknown          6 building_structure
1 sky                   7 infrastructure
2 water                 8 human
3 terrain               9 animal
4 road_paved           10 vehicle
5 vegetation           11 prop
```

Every window reads 124 consecutive pairs beginning at its `start_frame`. Qwen
samples those references every 12 frames. This repository consumes prepared
condition files; it does not generate them from an RGB video.

The RGB anchor may have a different resolution, but its aspect ratio must be
between 1:4 and 4:1. It is resized independently to short edge 2048 for the
Picture reference and to 1344x768 for the fixed first target frame.

Each `windows` entry provides one prompt and the first condition-frame ordinal
for that window. Successive `start_frame` values must advance by exactly 90.
The final length is:

```text
124 + 90 * (window_count - 1) frames
```

Caption line endings are canonicalized to CRLF before Qwen encoding so that
editors normalizing CRLF to LF do not change the consumed tokens.

The bundled first-window and continuation system prompts are selected internally
by window index. The configuration and command-line interface do not accept a
system-prompt override.

## Configuration

Set the paths, per-window prompts, start frames, and seed in your copied JSON
file. Keep these tested inference settings:

```json
{
  "steps": 20,
  "attention_mode": "flash3",
  "blocks_to_swap": 48,
  "expected_lora_modules": 200
}
```

`blocks_to_swap` may be reduced when enough GPU memory is available. The other
listed settings are fixed by this release.

## Run inference

From the prepared `release` directory, select a released configuration without
changing its inputs, prompts, or seed:

```bash
CONFIG=examples/example_01/config.json
```

Validation checks the configuration and verifies that all configured model and
input paths exist. It does not load weights or run a GPU job:

```bash
python -m cwm_h3_inference validate --config "$CONFIG"
```

Prepare the condition, anchor, first-frame latent, and Qwen caches:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cwm_h3_inference prepare --config "$CONFIG"
```

Generate the windows, continue from valid per-window recovery files when
present, stitch the sequence, and write the final MP4 and JSON receipt:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cwm_h3_inference generate --config "$CONFIG"
```

Use a new, absent `cache_dir` and output path whenever the model, LoRA, anchor,
condition input, prompt, start frame, or seed changes, then run `prepare` again.
Do not copy caches or recovery files from an earlier run into those paths, and
do not run two processes against the same cache and output paths. Generation
refuses to overwrite an existing final MP4.

A released multi-window run writes `124 + 90 * (window_count - 1)` frames at
1344x768 and 24 FPS. The retained single-window smoke-test template writes a
silent 124-frame video with the same resolution and frame rate.

The compact Hugging Face dataset is source input, not a runnable cache bundle.
Do not point inference directly at its `condition/*.npz` files. Run
`scripts/install_release_assets.sh` once, then run `prepare` to create fresh
Safetensors caches before `generate`.
