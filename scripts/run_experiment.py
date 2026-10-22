#!/usr/bin/env python3
"""Run a checked-in experiment configuration, relative to the repository root."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    raw = sys.argv[1:]
    split = raw.index('--') if '--' in raw else len(raw)
    args = parser.parse_args(raw[:split])
    extra = raw[split + 1:] if split < len(raw) else []
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    if 'script' in config:
        command = [sys.executable, str(ROOT / config['script']), *config['args']]
    elif 'module' in config:
        command = [sys.executable, '-m', config['module'], *config.get('args', [])]
    elif path.name.startswith('hamiltonian_'):
        command = [sys.executable, '-m', 'experiments.hamiltonian.run', '--config', str(path)]
    elif path.name.startswith('affine_gpu_'):
        command = [sys.executable, '-m', 'experiments.affine_gpu.run', '--config', str(path)]
    else:
        parser.error('configuration must specify script+args or module+args')
    command.extend(extra)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    versions = {}
    for package in ['numpy', 'scipy', 'matplotlib', 'torch', 'jax', 'matfree', 'gpytorch', 'linear_operator']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        config_label = str(path.relative_to(ROOT))
    except ValueError:
        config_label = str(path)
    record = {'config': config_label, 'command': command[1:],
              'started_utc': timestamp, 'python': platform.python_version(),
              'platform': platform.platform(), 'versions': versions,
              'thread_environment': {k: env.get(k) for k in
                  ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}}
    code = subprocess.run(command, cwd=ROOT, env=env).returncode
    record.update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
    out = ROOT / 'outputs/run_records' / (path.stem + '-' + timestamp + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print('Run record:', out)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
