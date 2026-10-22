#!/usr/bin/env python3
"""Regenerate all eight figures and eight tables from the frozen paper records."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    for ext in ['png', 'pdf']:
        commands.append(['experiments/matrix_log/plot.py', 'results/paper/matrix_log/results.json',
                         '--out', str(output / 'matrix_log' / ('figure1.' + ext))])
    commands += [
        ['experiments/gp_training/plot.py', '--outdir', str(output / 'gp_training')],
        ['-m', 'experiments.graph.plots', '--out-dir', str(output / 'graph')],
        ['-m', 'experiments.hamiltonian.plot', '--output', str(output / 'hamiltonian')],
        ['-m', 'experiments.affine_gpu.plot', '--output', str(output / 'affine_gpu')],
    ]
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(ROOT / '.cache/matplotlib')}
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for command in commands:
        print('Generating:', ' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    outputs = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (output / 'manifest.json').write_text(json.dumps({
        'source': 'frozen numerical records in results/paper',
        'note': 'Figures/tables are regenerated; no experiment timings are remeasured by this command.',
        'files_sha256': outputs}, indent=2) + '\n')
    print(f'Regenerated {len(outputs)} files in {output}')


if __name__ == '__main__':
    main()
