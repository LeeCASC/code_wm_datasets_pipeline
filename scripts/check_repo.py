#!/usr/bin/env python3
"""Check vendored fingerprints, portable source paths, and Git asset exclusions."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'third_party/files.sha256.json').read_text())
    for row in manifest:
        path = ROOT/'third_party'/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Vendored source differs from recorded snapshot: {row["path"]}')
    if any((ROOT/'third_party'/name/'.git').exists() for name in ('code_world_model', 'musubi_tuner')):
        raise ValueError('Unexpected nested .git directory')
    for folder in ('src', 'configs'):
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in ('.py', '.json'):
                contents = path.read_text()
                if '/home/lizihao/' in contents or '/datacc05/' in contents:
                    raise ValueError(f'Machine-specific source path: {path.relative_to(ROOT)}')
    candidates = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    names = set(filter(None, candidates.decode().split('\0')))
    for name in names:
        path = ROOT/name
        if path.suffix in ('.safetensors', '.mp4', '.f32', '.wav'):
            raise ValueError(f'Runtime asset unexpectedly eligible for Git: {name}')
        if path.is_file() and path.stat().st_size > 25*1024*1024:
            raise ValueError(f'Unexpected large Git file: {name}')
    models = json.loads((ROOT/'configs/model_manifest.json').read_text())
    for row in models['files']:
        path = ROOT/'models'/row['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes']:
            raise ValueError(f'Missing/nonregular model copy: {row["path"]}')
        result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', str(path)])
        if result.returncode:
            raise ValueError(f'Model file is not Git-ignored: {row["path"]}')
    report = {'status': 'pass', 'vendored_files_verified': len(manifest),
              'git_candidate_files': len(names), 'git_candidate_bytes': sum((ROOT/name).stat().st_size for name in names if (ROOT/name).is_file()),
              'git_ignored_model_files': len(models['files']), 'model_bytes': models['total_bytes'],
              'remote_push_performed': False}
    (ROOT/'reports/repository_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
