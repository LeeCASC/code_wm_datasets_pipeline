"""Small shared file helpers."""
import json
from pathlib import Path
import uuid


def read_json(path, fallback=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else fallback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
