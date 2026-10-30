"""Repository paths, independent of the caller's working directory."""
from pathlib import Path
import os
import sys

ROOT = Path(os.environ.get('WORLD_MODEL_PIPELINE_ROOT', Path(__file__).resolve().parents[2])).resolve()


def enable_vendored_sources():
    for name in ('musubi_tuner', 'code_world_model'):
        source = ROOT / 'third_party' / name / 'src'
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
