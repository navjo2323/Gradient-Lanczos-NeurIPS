"""Fresh CPU audit; never modifies the original source or recorded CSVs.

Uses the current graph implementation's numerical kernels, omits its unused
exact Lanczos adjoint, and times complete value plus projected gradient work.
The output is a new experiment, not a reconstruction of historical timings.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import scipy.linalg as la
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path('/Users/nsingh2/Desktop/Projects/Gradients-Lanczos/Graph-estrada-benchmark')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'fAb-frechet'))
import benchmark_pkg_lanczos as kernels
from benchmark_pkg_directions import NodeOuterDirection, OnesOuterDirection
from benchmark_pkg_graph_io import load_graph_matrix, unit_vector
from paper_benchmark import ko_bilinear_symmetric_safe


def evaluate_complete(A, direction, u, v, depth):
    start = perf_counter()
    # Match the original theta=0, time=1 preparation, including allocations.
    explicit = direction.maybe_matrix()
    A_theta = A.copy() if explicit is None else (A + 0.0 * explicit).tocsr()
    At = (1.0 * A_theta).tocsr()
    starts = np.column_stack([u + v, u - v])
    forward_start = perf_counter()
    out = kernels.lanczos_forward_batched(lambda X: At @ X, starts, depth)
    forward_seconds = perf_counter() - forward_start
    values, gradients = [], []
    small_seconds, contraction_seconds = 0.0, 0.0
    for col in range(2):
        effective_depth = out.K_eff[col]
        basis = [block[:, col] for block in out.x]
        small_start = perf_counter()
        T = kernels.build_tridiag(out.a[col, :effective_depth], out.b[col, :effective_depth])
        e1 = np.zeros(effective_depth)
        e1[0] = 1.0
        weight = float(out.norm_v[col]) ** 2
        value = weight * float(e1 @ la.expm(T) @ e1)
        G = kernels.small_matrix_gradient_exp(T, weight)
        small_seconds += perf_counter() - small_start
        contraction_start = perf_counter()
        projected_direction = direction.projected_from_basis(basis, effective_depth)
        gradient = float(np.sum(G * projected_direction))
        contraction_seconds += perf_counter() - contraction_start
        values.append(value)
        gradients.append(gradient)
    value = 0.25 * (values[0] - values[1])
    gradient = 0.25 * (gradients[0] - gradients[1])
    complete_seconds = perf_counter() - start
    return {
        'value': value, 'gradient': gradient,
        'complete_seconds': complete_seconds,
        'historical_scope_seconds': forward_seconds + contraction_seconds,
        'small_matrix_seconds': small_seconds,
        'effective_depths': out.K_eff,
        'start_norms': out.norm_v.tolist(),
    }


def recorded_rows(graph, objective, depth):
    filename = ROOT / f'results_paper_batched_safe_r3_{graph}_{objective}.csv'
    with filename.open() as f:
        return [r for r in csv.DictReader(f) if int(r['k']) == depth]


def measure_ours(A, direction, u, v, depth):
    start = perf_counter()
    result = evaluate_complete(A, direction, u, v, depth)
    elapsed = perf_counter() - start
    result['internal_complete_seconds'] = result['complete_seconds']
    result['complete_seconds'] = elapsed
    return result


def measure_ko(A, direction, u, v, depth):
    start = perf_counter()
    result = ko_bilinear_symmetric_safe(A, direction, 0.0, u, v, depth, 1.0)
    return result, perf_counter() - start


def main():
    report = {
        'status': 'fresh_run_current_source_not_historical_timing',
        'platform': platform.platform(), 'python': sys.version,
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'requested_threadpool_limit': 1,
        'thread_environment': {k: os.environ.get(k) for k in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
        'warmup_per_case_and_depth': 1,
        'measured_repeats': 3,
        'timing_scope': 'Each method complete forward value and gradient; reference, graph loading, warmup excluded. Ours excludes unused exact Lanczos adjoint.',
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['benchmark_pkg_lanczos.py', 'paper_benchmark.py',
                                       'benchmark_pkg_references.py', 'benchmark_pkg_directions.py',
                                       'fAb-frechet/lib.py']},
        'cases': [],
    }
    with threadpool_limits(limits=1):
        report['threadpools'] = threadpool_info()
        for graph, filename, i, j in [('hepth', 'ca-HepTh.txt', 7200, 6969),
                                      ('amazon', 'com-amazon.ungraph.txt', 0, 53525)]:
            A = load_graph_matrix(str(ROOT / 'graphs' / filename), symmetrize=True)
            u, v = unit_vector(A.shape[0], i), unit_vector(A.shape[0], j)
            for objective in ['tn', 'sc']:
                direction = OnesOuterDirection(n=A.shape[0]) if objective == 'tn' else NodeOuterDirection(n=A.shape[0], ell=i)
                for depth in [16, 32]:
                    old = recorded_rows(graph, objective, depth)
                    measure_ours(A, direction, u, v, depth)
                    measure_ko(A, direction, u, v, depth)
                    repeats = []
                    for repetition in range(3):
                        if repetition % 2:
                            ko, ko_seconds = measure_ko(A, direction, u, v, depth)
                            ours = measure_ours(A, direction, u, v, depth)
                        else:
                            ours = measure_ours(A, direction, u, v, depth)
                            ko, ko_seconds = measure_ko(A, direction, u, v, depth)
                        repeats.append({'ours': ours, 'ko_seconds': ko_seconds,
                                        'ko_internal_seconds': ko.runtime_sec,
                                        'ko_value': ko.value, 'ko_gradient': ko.grad,
                                        'ko_kind': ko.kind,
                                        'complete_speedup': ko_seconds / ours['complete_seconds']})
                    ov, og = float(old[0]['ours_value']), float(old[0]['ours_grad'])
                    kv, kg = float(old[0]['ma_value']), float(old[0]['ma_grad'])
                    row = {'graph': graph, 'objective': objective, 'depth': depth,
                           'n': A.shape[0], 'nnz': A.nnz, 'i': i, 'j': j, 'ell': i if objective == 'sc' else 0,
                           'recorded_csv': str(ROOT / f'results_paper_batched_safe_r3_{graph}_{objective}.csv'),
                           'recorded_ours_value': ov, 'recorded_ours_gradient': og,
                           'repeats': repeats,
                           'mean_complete_speedup': float(np.mean([r['complete_speedup'] for r in repeats])),
                           'mean_ours_seconds': float(np.mean([r['ours']['complete_seconds'] for r in repeats])),
                           'mean_ko_seconds': float(np.mean([r['ko_seconds'] for r in repeats])),
                           'mean_small_matrix_seconds': float(np.mean([r['ours']['small_matrix_seconds'] for r in repeats])),
                           'max_ours_value_difference_vs_recorded': max(abs(r['ours']['value'] - ov) / max(1.0, abs(ov)) for r in repeats),
                           'max_ours_gradient_difference_vs_recorded': max(abs(r['ours']['gradient'] - og) / max(1.0, abs(og)) for r in repeats),
                           'max_ko_value_difference_vs_recorded': max(abs(r['ko_value'] - kv) / max(1.0, abs(kv)) for r in repeats),
                           'max_ko_gradient_difference_vs_recorded': max(abs(r['ko_gradient'] - kg) / max(1.0, abs(kg)) for r in repeats)}
                    report['cases'].append(row)
                    print(json.dumps({k: row[k] for k in ['graph', 'objective', 'depth', 'mean_complete_speedup', 'mean_ours_seconds', 'mean_ko_seconds', 'mean_small_matrix_seconds', 'max_ours_value_difference_vs_recorded', 'max_ours_gradient_difference_vs_recorded']}), flush=True)
                    Path(__file__).with_name('fresh_timing_results.json').write_text(json.dumps(report, indent=2))
            del A


if __name__ == '__main__':
    main()
