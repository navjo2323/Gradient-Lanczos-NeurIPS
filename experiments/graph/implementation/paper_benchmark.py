from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import numpy as np

from benchmark_pkg_directions import IdentityDirection, NodeOuterDirection, OnesOuterDirection
from benchmark_pkg_graph_io import (
    apply_matrix_transform,
    choose_existing_edge,
    induced_subgraph,
    load_graph_matrix,
    unit_vector,
)
from benchmark_pkg_lanczos import (
    evaluate_bilinear_with_both_gradients,
    evaluate_quadratic_with_both_gradients,
)
from benchmark_pkg_references import (
    ReferenceEval,
    exact_dense_quadratic,
    ko_bilinear,
    ko_quadratic,
    sparse_block_quadratic,
)


def rel_err(x: float, y: float) -> float:
    return abs(x - y) / max(1.0, abs(y))


def append_csv_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def ko_bilinear_symmetric_safe(A, direction, theta, u, v, k, time_scale) -> ReferenceEval:
    """Call KO's bilinear routine once, avoiding sep_orth breakdown when E @ v = 0.

    For symmetric A and symmetric direction E,
        u^T L_exp(A,E) v = v^T L_exp(A,E) u.
    KO's separate-orthogonalization implementation starts from b=v and forms
    X_1 from E b. If E v is zero but E u is nonzero, swapping the two test
    vectors computes the same scalar while giving KO a nonzero derivative start.
    """
    if np.linalg.norm(direction.apply(v)) > 1e-14:
        return ko_bilinear(A, direction, theta, u, v, k, time_scale)
    if np.linalg.norm(direction.apply(u)) > 1e-14:
        swapped = ko_bilinear(A, direction, theta, v, u, k, time_scale)
        return ReferenceEval(
            value=swapped.value,
            grad=swapped.grad,
            runtime_sec=swapped.runtime_sec,
            kind=f"{swapped.kind}_swapped_start",
        )
    raise RuntimeError(
        "KO sep_orth breaks down because E @ u and E @ v are both zero. "
        "Choose an SC endpoint equal to ell, or use a block-Arnoldi fallback."
    )


def bilinear_reference(args, A, direction, u, v) -> Optional[ReferenceEval]:
    if A.shape[0] <= args.dense_check_max_n:
        rp = exact_dense_quadratic(A, direction, args.theta, u + v, args.time)
        rm = exact_dense_quadratic(A, direction, args.theta, u - v, args.time)
    elif args.sparse_block_reference:
        rp = sparse_block_quadratic(A, direction, args.theta, u + v, args.time)
        rm = sparse_block_quadratic(A, direction, args.theta, u - v, args.time)
    else:
        return None

    return ReferenceEval(
        value=0.25 * (rp.value - rm.value),
        grad=0.25 * (rp.grad - rm.grad),
        runtime_sec=rp.runtime_sec + rm.runtime_sec,
        kind=rp.kind,
    )


def quadratic_reference(args, A, direction, u) -> Optional[ReferenceEval]:
    if A.shape[0] <= args.dense_check_max_n:
        return exact_dense_quadratic(A, direction, args.theta, u, args.time)
    if args.sparse_block_reference:
        return sparse_block_quadratic(A, direction, args.theta, u, args.time)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-style benchmark: compare OUR APPROX gradient vs Modified Arnoldi."
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--paper-objective",
        choices=["tn", "sc", "ei"],
        default="tn",
        help=(
            "tn = total network communicability sensitivity, "
            "sc = subgraph centrality sensitivity, "
            "ei = Estrada index sensitivity."
        ),
    )
    parser.add_argument(
        "--matrix",
        choices=["raw_adj", "norm_adj", "neg_norm_laplacian"],
        default="raw_adj",
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--k-values", type=str, default=None)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=1)
    parser.add_argument("--ell", type=int, default=0)
    parser.add_argument("--subgraph-size", type=int, default=None)
    parser.add_argument("--dense-check-max-n", type=int, default=200)
    parser.add_argument("--sparse-block-reference", action="store_true")
    parser.add_argument("--compare-kressner", action="store_true")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    A_raw = load_graph_matrix(args.graph, symmetrize=True)
    if args.subgraph_size is not None:
        A_raw = induced_subgraph(A_raw, args.subgraph_size)

    n = A_raw.shape[0]
    if args.i == 0 and args.j == 1:
        args.i, args.j = choose_existing_edge(A_raw)
    args.i = min(args.i, n - 1)
    args.j = min(args.j, n - 1)
    args.ell = min(args.ell, n - 1)

    A = apply_matrix_transform(A_raw, args.matrix)

    if args.paper_objective == "tn":
        direction = OnesOuterDirection(n=n)
        mode = "bilinear"
        u = unit_vector(n, args.i)
        v = unit_vector(n, args.j)

    elif args.paper_objective == "sc":
        direction = NodeOuterDirection(n=n, ell=args.ell)
        mode = "bilinear"
        u = unit_vector(n, args.i)
        v = unit_vector(n, args.j)

    elif args.paper_objective == "ei":
        direction = IdentityDirection(n=n)
        mode = "bilinear"
        u = unit_vector(n, args.i)
        v = unit_vector(n, args.j)

    else:
        raise ValueError(args.paper_objective)

    ks = [args.k] if args.k_values is None else [int(tok) for tok in args.k_values.split(",") if tok.strip()]

    print(f"Loaded symmetric graph with n={n}, nnz={A_raw.nnz}.")
    print(f"Paper objective={args.paper_objective}, matrix={args.matrix}, t={args.time}.")
    print(f"Indices: i={args.i}, j={args.j}, ell={args.ell}.")

    for k in ks:
        print("-" * 72)
        print(f"k = {k}")

        if mode == "bilinear":
            ours = evaluate_bilinear_with_both_gradients(A, direction, args.theta, u, v, k, args.time)
            ref = bilinear_reference(args, A, direction, u, v)
            ko = None

            if args.compare_kressner:
                try:
                    ko = ko_bilinear_symmetric_safe(A, direction, args.theta, u, v, k, args.time)
                except Exception as exc:
                    print(f"Modified Arnoldi unavailable: {exc}")

        else:
            ours = evaluate_quadratic_with_both_gradients(A, direction, args.theta, u, k, args.time)
            ref = quadratic_reference(args, A, direction, u)
            ko = None
            if args.compare_kressner:
                try:
                    ko = ko_quadratic(A, direction, args.theta, u, k, args.time)
                except Exception as exc:
                    print(f"Modified Arnoldi unavailable: {exc}")

        ours_runtime = ours.runtime_forward_sec + ours.runtime_approx_grad_sec

        print(f"Ours value:                   {ours.value:.12e}")
        print(f"Ours approx grad:             {ours.grad_approx:.12e}")
        print(f"Ours runtime total (sec):     {ours_runtime:.6f}")
        print(f"Ours runtime forward (sec):   {ours.runtime_forward_sec:.6f}")
        print(f"Ours runtime approx  (sec):   {ours.runtime_approx_grad_sec:.6f}")
        print(f"Ours K_eff:                   {ours.k_eff}")

        if ref is not None:
            print(f"Reference ({ref.kind}) value: {ref.value:.12e}")
            print(f"Reference ({ref.kind}) grad:  {ref.grad:.12e}")
            print(f"Ours value error vs ref:      {rel_err(ours.value, ref.value):.3e}")
            print(f"Ours grad error vs ref:       {rel_err(ours.grad_approx, ref.grad):.3e}")
        else:
            print("No reference computed.")

        if ko is not None:
            print(f"Modified Arnoldi value:       {ko.value:.12e}")
            print(f"Modified Arnoldi grad:        {ko.grad:.12e}")
            print(f"Modified Arnoldi runtime:     {ko.runtime_sec:.6f}")
            if ref is not None:
                print(f"Modified Arnoldi value err:   {rel_err(ko.value, ref.value):.3e}")
                print(f"Modified Arnoldi grad err:    {rel_err(ko.grad, ref.grad):.3e}")
            print(f"Ours-vs-ModifiedArnoldi grad: {rel_err(ours.grad_approx, ko.grad):.3e}")

        if args.csv_out:
            row = {
                "graph": args.graph,
                "paper_objective": args.paper_objective,
                "matrix": args.matrix,
                "n": n,
                "nnz": A_raw.nnz,
                "k": k,
                "time": args.time,
                "theta": args.theta,
                "i": args.i,
                "j": args.j,
                "ell": args.ell,
                "ours_value": ours.value,
                "ours_grad": ours.grad_approx,
                "ours_runtime_total": ours_runtime,
                "ours_runtime_forward": ours.runtime_forward_sec,
                "ours_runtime_approx": ours.runtime_approx_grad_sec,
                "ours_k_eff": ours.k_eff,
                "reference_kind": None if ref is None else ref.kind,
                "reference_value": None if ref is None else ref.value,
                "reference_grad": None if ref is None else ref.grad,
                "ours_value_err": None if ref is None else rel_err(ours.value, ref.value),
                "ours_grad_err": None if ref is None else rel_err(ours.grad_approx, ref.grad),
                "ma_value": None if ko is None else ko.value,
                "ma_grad": None if ko is None else ko.grad,
                "ma_runtime": None if ko is None else ko.runtime_sec,
                "ma_value_err": None if (ko is None or ref is None) else rel_err(ko.value, ref.value),
                "ma_grad_err": None if (ko is None or ref is None) else rel_err(ko.grad, ref.grad),
            }
            append_csv_row(args.csv_out, row)


if __name__ == "__main__":
    main()
