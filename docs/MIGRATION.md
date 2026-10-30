# Source migration

The original working directory, model originals, datasets, environments and experiment outputs remain unchanged. This directory is an independent source checkout with copied model files.

| Original file | New module |
| --- | --- |
| `aircraft_proxy.py` | `geometry/aircraft.py` |
| `scripts/make_aircraft_box_proxy.py` | `geometry/obb.py` |
| `scripts/cwm_a100.py` | `inference/cwm.py` |
| `scripts/run_h3_rgb_reference.py` | `inference/rgb.py` |
| `scripts/prepare_cwm_prompt_suite.py` | `workflows/aircraft.py` |
| `scripts/prepare_observer_ref2va.py` | `workflows/observers.py` |
| `scripts/run_cwm_prompt_suite.py` | `orchestration/aircraft.py` |
| `scripts/run_observer_ref2va_suite.py` | `orchestration/observers.py` |
| `scripts/run_aircraft_cwm.py` | Offline `orchestration/supervisor.py`; implicit network downloads removed |
| `scripts/audit_cwm_geometry.py` | `diagnostics/geometry.py` |
| `scripts/download_direct_file.py` | `tools/download.py` |
| `scripts/convert_local_h3.py` | `tools/convert.py` |

Additional shared code owns path resolution, model paths, offline/proxy-free subprocess environments, GPU selection, atomic JSON output, and stream-copy concatenation. New CLI: `python scripts/wmp --help` or installed `wmp`.

The standalone download and conversion tools remain available as `python -m world_model_pipeline.tools.download --help` and `python -m world_model_pipeline.tools.convert --help` after editable installation. The converter requires an explicit local source and uses the bundled structure headers. Normal inference does not need conversion or downloading because its assets have been copied.

CWM and Musubi are the only external source repositories used by these executed routes. PWM, Code-as-World, Matrix-Game, Procedura and elastic-forcing were not copied into this runnable subset because these experiments did not execute them. Their original directories remain available outside this new checkout.

Example PNG anchors were copied, while prior large caches and full experiment output trees were not made source dependencies. Git ignores local `models`, `outputs`, `data`, caches and environments. Model hashes and third-party revisions are versioned separately from the large artifacts.
