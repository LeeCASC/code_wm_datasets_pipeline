# Validation

This file records the verification of the refactor, rather than claiming a new neural inference result.

- Existing numerical regression tests were migrated to package imports: projection/depth conventions, visibility, motion/contact, OBB bounds, cache reuse restrictions, GPU eligibility, camera conversion, source-frame mapping, FP32/FP16 decoder compatibility, direct download SHA checks and VAE conversion.
- Additional checks cover running the CLI outside the checkout, importing the copied third-party sources, portable model/config paths, offline/proxy-free worker environments, and stream-copy concatenation with frame/timestamp/boundary-pixel verification.
- Each model file is copied into a fresh regular file, read back for SHA-256 equality, and recorded in `configs/model_manifest.json`. The detailed local receipt is `reports/model_copy.json`.

## Checks run on 2026-10-30

- `python -m pytest -q`: **33 passed**, 2.48 seconds in the existing validated Python environment. This includes a regression for keeping the requested appearance anchor in all nine photo-conditioned jobs and using the box anchor only in the explicit tenth control.
- The new CLI rendered a 214-frame side-view aircraft proxy and prepared all 10 prompt/OBB configurations. Six sampled source depth/semantic files are byte-identical to their previous-run counterparts.
- The new CLI prepared both UE captures into four 124-frame reference segments. All 496 camera-mapping records, including overlap, exactly match the prior implementation. All four copied appearance anchors are present.
- Stream-copy concatenation produced a 338-frame / 14.0833-second proxy+OBB preview, checking the full timestamp sequence and four source-boundary pixel hashes.
- Native Musubi H3 and CWM engine import probes resolved to the new `third_party/` directories, including a probe with CUDA hidden.
- Wheel and source archive builds succeeded and were inspected for vendored source inclusion and absence of model weights.
- All 10 CWM configs pass the actual upstream file/config validator. All four native RGB configs reference existing local models and prepared inputs.
- All 14 model/processor files, totaling 125,996,929,736 bytes, passed copy-time SHA-256 read-back verification and are independent regular files. Git exclusions were checked against every file.

Local records are in `reports/workflow_smoke.json`, `reports/package_check.json`, `reports/model_copy.json`, and `reports/repository_check.json`. This refactor does not itself create a new GPU inference or neural-quality evaluation.
