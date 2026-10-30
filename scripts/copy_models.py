#!/usr/bin/env python3
"""Copy local H3 assets as independent files and verify every copied byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy(source, project):
    destination = project / 'models'
    files = sorted(p for p in source.rglob('*') if p.is_file() and (
        p.suffix == '.safetensors' or
        ('processors' in p.relative_to(source).parts and p.name != 'download_receipt.json')
    ))
    if not files:
        raise FileNotFoundError(f'No model files found in {source}')
    total = sum(p.stat().st_size for p in files)
    state = {'phase': 'copying', 'total_bytes': total, 'completed_bytes': 0, 'files': []}
    state_path = project / 'reports/model_copy.json'
    started = time.monotonic()
    manifest = []
    for original in files:
        relative = original.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        temporary = target.with_suffix(target.suffix + '.copying')
        hasher = hashlib.sha256()
        amount = 0
        last = 0
        with original.open('rb') as reader, temporary.open('xb') as writer:
            while block := reader.read(16 * 1024 * 1024):
                writer.write(block)
                hasher.update(block)
                amount += len(block)
                if time.monotonic() - last > 3:
                    state.update(current_file=str(relative), current_bytes=amount,
                                 elapsed_seconds=round(time.monotonic()-started, 1))
                    write_json(state_path, state)
                    last = time.monotonic()
            writer.flush()
            os.fsync(writer.fileno())
        expected = hasher.hexdigest()
        state.update(phase='verifying', current_file=str(relative))
        write_json(state_path, state)
        actual = digest(temporary)
        if actual != expected or temporary.stat().st_size != original.stat().st_size:
            raise ValueError(f'Copy verification failed: {relative}')
        if temporary.stat().st_ino == original.stat().st_ino and temporary.stat().st_dev == original.stat().st_dev:
            raise ValueError('Copy unexpectedly shares the source inode')
        temporary.replace(target)
        row = {'path': str(relative), 'bytes': amount, 'sha256': actual}
        manifest.append(row)
        state['files'].append(dict(row, verified=True, independent_file=True))
        state['completed_bytes'] += amount
        state['phase'] = 'copying'
        write_json(state_path, state)
        print(json.dumps({'copied': str(relative), 'bytes': amount, 'sha256': actual}), flush=True)
    write_json(project/'configs/model_manifest.json', {
        'schema': 'local_h3_model_manifest_v1', 'total_bytes': total, 'files': manifest,
        'weights_in_git': False,
        'note': 'These hashes identify local converted/shared H3 components, not bytewise parity with every upstream package.'})
    state.update(phase='complete', elapsed_seconds=round(time.monotonic()-started, 1))
    write_json(state_path, state)
    print(json.dumps({'phase': 'complete', 'bytes': total, 'files': len(manifest)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    copy(args.source.resolve(), args.project.resolve())
