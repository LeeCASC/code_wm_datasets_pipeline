


<div align="center">

<h1>Code World Model:<br>Coding Agent as World Brain</h1>

<p>
  <a href="https://buaacyw.github.io/"><strong>Yiwen Chen</strong></a><sup>1,2</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a><sup>2</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://icoz69.github.io/"><strong>Chi Zhang</strong></a><sup>1</sup>
</p>

<p>
  <sup>1</sup> AGI Lab, Westlake University
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sup>2</sup> Nanyang Technological University
</p>

<p>
  <a href="https://buaacyw.github.io/cwm/"><strong>Project Page</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2608.25927"><strong>arXiv</strong></a>
</p>

</div>

---

https://github.com/user-attachments/assets/4c13c5fb-ab45-4a0e-9e59-b85499ef7390

## Quick start

1. Follow the [environment and installation guide](ENVIRONMENT.md).
2. From the repository root, download the public
   [CWM LoRA](https://huggingface.co/NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints)
   and deterministically expand the
   [40 compact Hugging Face examples](https://huggingface.co/datasets/NTU-yiwen/code-world-model-inference-examples-40):

```bash
./scripts/install_release_assets.sh "$PWD/release"
```

3. Enter the prepared release directory and run a released configuration
   without changing its inputs, prompts, or seed:

```bash
cd release
CONFIG=examples/example_01/config.json
python -m cwm_h3_inference validate --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 python -m cwm_h3_inference prepare --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 python -m cwm_h3_inference generate --config "$CONFIG"
```

Use a new cache directory and output path for every changed experiment.
The Hugging Face examples contain compact NPZ conditions, not caches or output
videos; the installation script performs the required local format expansion.

## Documentation

- [Environment, installation, model, LoRA, and example downloads](ENVIRONMENT.md)
- [Input format, configuration, and inference](INFERENCE.md)
- [License](LICENSE), [project notice](NOTICE), and
  [third-party notices](THIRD_PARTY_NOTICES.md)
