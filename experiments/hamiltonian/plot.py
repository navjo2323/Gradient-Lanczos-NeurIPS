"""Regenerate Figure 6 and Table 2 directly from the saved training histories."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summarize(record):
    dense = record['dense_history'][-1]
    rows = [{'method': 'Dense', 'depth': '', 'initial_gradient_relative_error': None,
             'final_parameter_error': dense['param_error'], 'evaluation_seconds': dense['wall_time'], 'speedup': 1.0}]
    for depth, run in record['gm_runs'].items():
        last = run['history'][-1]
        rows.append({'method': f'm={depth}', 'depth': int(depth),
                     'initial_gradient_relative_error': run['initial_gradient_diagnostic']['grad_rel_error'],
                     'final_parameter_error': last['param_error'], 'evaluation_seconds': last['wall_time'],
                     'speedup': dense['wall_time'] / last['wall_time']})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results/paper/hamiltonian/results.json')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/hamiltonian')
    args = p.parse_args()
    data = json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summarize(data)
    with (args.output / 'table_2.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    def sci(x):
        mantissa, exponent = f'{x:.2e}'.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Method & Grad. rel. err. & Final param. error & Speedup \\', r'\midrule']
    for r in rows:
        error = '--' if r['initial_gradient_relative_error'] is None else sci(r['initial_gradient_relative_error'])
        lines.append(f"{r['method']} & {error} & {sci(r['final_parameter_error'])} & ${r['speedup']:.1f}\\times$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (args.output / 'table_2.tex').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    series = [('Dense', data['dense_history'])] + [(f'Projected $m={d}$', r['history']) for d, r in data['gm_runs'].items()]
    for label, history in series:
        for ax, key in zip(axes, ['loss', 'param_error']):
            ax.semilogy([r['step'] for r in history], [r[key] for r in history], label=label)
    for ax, title, ylabel in zip(axes, ['(a) Training loss', '(b) Parameter recovery'], ['Loss', r'$\|\theta-\theta_\star\|_2$']):
        ax.set(xlabel='Adam updates', ylabel=ylabel, title=title)
        ax.grid(True, alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(args.output / f'figure_6.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote Figure 6 and Table 2 to {args.output}')


if __name__ == '__main__':
    main()
