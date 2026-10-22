"""Portable graph accuracy and complete-call timing reproduction.

The numerical kernel and timer body are preserved from the corrected paper audit.
New runs are written separately from the frozen paper records.
"""
from __future__ import annotations
import argparse
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
import scipy.sparse as sp
from threadpoolctl import threadpool_info, threadpool_limits
from ._bootstrap import ROOT
from .data import DATASETS, verify
import benchmark_pkg_lanczos as kernels
from benchmark_pkg_directions import NodeOuterDirection, OnesOuterDirection
from benchmark_pkg_graph_io import load_graph_matrix, unit_vector
from benchmark_pkg_references import sparse_block_quadratic
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


def scaled_error(value, reference):
    return abs(value - reference) / max(1.0, abs(reference))


def recorded_row(graph, objective, depth):
    path = ROOT / "results/paper/graph/accuracy" / f"results_paper_batched_safe_{graph}_{objective}.csv"
    with path.open() as handle:
        return next(row for row in csv.DictReader(handle) if int(row["k"]) == depth)


def sparse_reference(A, direction, u, v):
    plus = sparse_block_quadratic(A, direction, 0.0, u + v, 1.0)
    minus = sparse_block_quadratic(A, direction, 0.0, u - v, 1.0)
    return 0.25 * (plus.value - minus.value), 0.25 * (plus.grad - minus.grad)


def smoke():
    """Check TN and SC polarization against dense exponential Fréchet derivatives."""
    rng = np.random.default_rng(284)
    dense = rng.uniform(0.0, 0.15, (12, 12))
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0.0)
    A = sp.csr_matrix(dense)
    u, v = unit_vector(12, 2), unit_vector(12, 7)
    checks = []
    for objective in ["tn", "sc"]:
        direction = OnesOuterDirection(12) if objective == "tn" else NodeOuterDirection(12, 2)
        E = np.ones((12, 12)) if objective == "tn" else np.outer(u, u)
        expected_value = float(u @ la.expm(dense) @ v)
        expected_gradient = float(u @ la.expm_frechet(dense, E, compute_expm=False) @ v)
        ours = measure_ours(A, direction, u, v, 10)
        ko, _ = measure_ko(A, direction, u, v, 10)
        ref_value, ref_gradient = sparse_reference(A, direction, u, v)
        errors = {
            "ours_value": scaled_error(ours["value"], expected_value),
            "ours_gradient": scaled_error(ours["gradient"], expected_gradient),
            "ko_value": scaled_error(ko.value, expected_value),
            "ko_gradient": scaled_error(ko.grad, expected_gradient),
            "sparse_reference_value": scaled_error(ref_value, expected_value),
            "sparse_reference_gradient": scaled_error(ref_gradient, expected_gradient),
        }
        if not all(np.isfinite(error) and error < 1e-9 for error in errors.values()):
            raise AssertionError(f"Dense oracle failed for {objective}: {errors}")
        checks.append({"objective": objective, "errors": errors, "ko_kind": ko.kind})
    return {"status": "passed", "seed": 284, "n": 12, "depth": 10, "tolerance": 1e-9, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "accuracy", "timing", "both"], default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/graph_paper.json")
    parser.add_argument("--graphs", nargs="+", choices=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/graphs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/generated/graph")
    parser.add_argument("--reference", choices=["stored", "recompute"], default="stored",
                        help="Recompute sparse block references, or compare with frozen paper references.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.mode == "smoke":
            report = smoke()
            (args.out_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return
        config = json.loads(args.config.read_text())
        fixed_settings = {"matrix": "raw_adj", "theta": 0.0, "time_scale": 1.0,
                          "precision": "float64", "reorthogonalization": "none",
                          "breakdown_tolerance": 1e-14}
        for key, value in fixed_settings.items():
            if config[key] != value:
                raise ValueError(f"The preserved paper kernel requires {key}={value!r}")
        if not set(config["objectives"]) <= {"tn", "sc"}:
            raise ValueError("Graph objectives must be tn or sc")
        if config["repeats"] < 2 or config["warmup_calls"] < 1:
            raise ValueError("Use at least one warm-up and two measured timing calls")
        metadata = {
            "platform": platform.platform(), "python": sys.version,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "requested_threadpool_limit": 1, "threadpools": threadpool_info(),
            "thread_environment": {key: os.environ.get(key) for key in
                ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
            "configuration": config, "graphs": args.graphs or config["graphs"],
            "reference_policy": args.reference,
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT / "experiments/graph").rglob("*.py"))},
        }
        timing = {"metadata": metadata, "cases": []}
        accuracy = []
        for graph in args.graphs or config["graphs"]:
            spec = DATASETS[graph]
            path = verify(graph, args.data_dir)
            A = load_graph_matrix(str(path), symmetrize=True)
            if A.shape != (spec["n"], spec["n"]) or A.nnz != spec["nnz"]:
                raise AssertionError(f"Loaded graph dimensions differ from the paper: {graph}")
            u, v = unit_vector(A.shape[0], spec["i"]), unit_vector(A.shape[0], spec["j"])
            for objective in config["objectives"]:
                direction = OnesOuterDirection(A.shape[0]) if objective == "tn" else NodeOuterDirection(A.shape[0], spec["i"])
                if args.mode in ["accuracy", "both"]:
                    computed_reference = sparse_reference(A, direction, u, v) if args.reference == "recompute" else None
                    for depth in config["accuracy_depths"]:
                        old = recorded_row(graph, objective, depth)
                        reference_value, reference_gradient = computed_reference or (float(old["reference_value"]), float(old["reference_grad"]))
                        ours = measure_ours(A, direction, u, v, depth)
                        ko, _ = measure_ko(A, direction, u, v, depth)
                        row = {"graph": graph, "objective": objective, "depth": depth,
                               "ours_value": ours["value"], "ours_gradient": ours["gradient"],
                               "ko_value": ko.value, "ko_gradient": ko.grad,
                               "reference_value": reference_value, "reference_gradient": reference_gradient}
                        for method, value, gradient, old_prefix in [
                            ("ours", ours["value"], ours["gradient"], "ours"), ("ko", ko.value, ko.grad, "ma")]:
                            row[f"{method}_value_error"] = scaled_error(value, reference_value)
                            row[f"{method}_gradient_error"] = scaled_error(gradient, reference_gradient)
                            row[f"{method}_value_difference_vs_recorded"] = scaled_error(value, float(old[f"{old_prefix}_value"]))
                            row[f"{method}_gradient_difference_vs_recorded"] = scaled_error(gradient, float(old[f"{old_prefix}_grad"]))
                        differences = [value for key, value in row.items() if key.endswith("difference_vs_recorded")]
                        if not all(np.isfinite(value) and value < config["recorded_match_tolerance"] for value in differences):
                            raise AssertionError(f"Paper numerical comparison failed: {row}")
                        accuracy.append(row)
                        print(json.dumps(row), flush=True)
                    with (args.out_dir / "accuracy.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(accuracy[0]))
                        writer.writeheader(); writer.writerows(accuracy)
                if args.mode in ["timing", "both"]:
                    for depth in config["timing_depths"]:
                        for _ in range(config["warmup_calls"]):
                            measure_ours(A, direction, u, v, depth)
                            measure_ko(A, direction, u, v, depth)
                        repeats = []
                        for repetition in range(config["repeats"]):
                            if repetition % 2:
                                ko, seconds = measure_ko(A, direction, u, v, depth)
                                ours = measure_ours(A, direction, u, v, depth)
                            else:
                                ours = measure_ours(A, direction, u, v, depth)
                                ko, seconds = measure_ko(A, direction, u, v, depth)
                            repeats.append({"ours": ours, "ko_seconds": seconds,
                                            "ko_value": ko.value, "ko_gradient": ko.grad,
                                            "complete_speedup": seconds / ours["complete_seconds"],
                                            "order": "ko,ours" if repetition % 2 else "ours,ko"})
                        timing["cases"].append({"graph": graph, "objective": objective,
                                                "depth": depth, "repeats": repeats})
                        (args.out_dir / "timings.json").write_text(json.dumps(timing, indent=2) + "\n")
                        print(json.dumps({"graph": graph, "objective": objective, "depth": depth,
                                          "mean_paired_speedup": float(np.mean([r["complete_speedup"] for r in repeats]))}), flush=True)
            del A
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
