"""Rerun selected paper depths against the archived dense reference.

This does not recompute the 15000-by-15000 dense eigendecomposition. It verifies
the released forward and gradient implementations at the paper's actual size.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

import run as experiment


def vector(values: dict) -> np.ndarray:
    return np.array([values[key] for key in ("log_lengthscale", "log_outputscale", "log_noise", "mean")], dtype=float)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, default=root / "results/paper/matrix_log/results.json")
    parser.add_argument("--depths", default="60,80")
    parser.add_argument("--modes", default="full,none", help="Comma-separated reorthogonalization modes")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if values or finite gradients differ from the archive beyond tolerance")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Relative forward / dense-norm-normalized gradient tolerance for --check")
    parser.add_argument("--output", type=Path, default=Path("outputs/matrix_log/paper_size_validation.json"))
    args = parser.parse_args()
    saved = json.loads(args.saved.read_text())
    config = experiment.Config(**saved["config"])
    depths = [int(m) for m in args.depths.split(",")]
    modes = args.modes.split(",")
    if not modes or any(mode not in ("full", "none") for mode in modes):
        raise ValueError("Modes must be full and/or none")
    archived = {row["m"]: row for row in saved["rows"]}
    if not set(depths) <= set(archived):
        raise ValueError("Requested depths must exist in the saved result file")
    experiment.jax.config.update("jax_enable_x64", True)
    data_key, probe_key = experiment.jax.random.split(experiment.jax.random.PRNGKey(config.seed))
    x, _ = experiment.make_toy_regression(data_key, config.n, config.d)
    params = experiment.init_params(dtype=x.dtype)
    probes = experiment.make_rademacher_probes(probe_key, config.num_probes, config.n, dtype=x.dtype)
    reference = experiment.GPParams(**{k: experiment.jnp.asarray(v, dtype=x.dtype) for k, v in saved["reference"]["grad"].items()})
    ref_norm = float(np.linalg.norm(vector(saved["reference"]["grad"])))
    output = {
        "description": "Fresh paper-size runs using the archived dense value/gradient reference; no new dense eigendecomposition",
        "config": dataclasses.asdict(config),
        "depths_rerun": depths,
        "reorthogonalization_modes_rerun": modes,
        "reference": saved["reference"],
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jax", "jaxlib", "matfree")},
        "comparison_tolerance": args.tolerance,
        "check_requested": args.check,
        "rows": [],
    }
    for depth in depths:
        for mode in modes:
            start = time.perf_counter()
            value, vs, ws = experiment.repo_forward_value_and_frozen_VW(params, x, probes, config, depth, reortho=mode)
            ours = experiment.ours_grad_from_frozen_VW(params, x, vs, ws, config)
            adj_value, adjoint = experiment.repo_adjoint_value_and_grad(params, x, probes, config, depth, reortho=mode)
            row = {
                "m": depth,
                "reorthogonalization": mode,
                "forward_value": value,
                "adjoint_forward_value": adj_value,
                "forward_relative_error_to_archived_dense": experiment.rel_scalar_error(value, saved["reference"]["value"]),
                "forward_relative_difference_from_archived_run": experiment.rel_scalar_error(value, archived[depth]["forward"][mode]["value"]),
                "runtime_sec": time.perf_counter() - start,
            }
            for name, grad in (("ours", ours), ("adjoint", adjoint)):
                gd = experiment.gp_to_dict(grad)
                old = archived[depth]["gradients"][f"{name}_{mode}"]["grad"]
                row[name] = {
                    "gradient": gd,
                    "relative_error_to_archived_dense": experiment.safe_rel_grad_error(grad, reference),
                    "difference_from_archived_gradient_normalized_by_dense_norm": float(np.linalg.norm(vector(gd) - vector(old)) / ref_norm),
                }
            differences = [row["forward_relative_difference_from_archived_run"]] + [
                row[name]["difference_from_archived_gradient_normalized_by_dense_norm"]
                for name in ("ours", "adjoint")
            ]
            row["matches_archive_within_tolerance"] = bool(all(np.isfinite(d) and d <= args.tolerance for d in differences))
            output["rows"].append(row)
            output["all_selected_cases_match_archive_within_tolerance"] = all(r["matches_archive_within_tolerance"] for r in output["rows"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(experiment._json_finite(output), indent=2, allow_nan=False) + "\n")
            print(json.dumps(experiment._json_finite(row), indent=2, allow_nan=False), flush=True)

    if args.check and not output["all_selected_cases_match_archive_within_tolerance"]:
        raise SystemExit("Selected measurements differ from the archive; see the output JSON and dependency-version notes.")


if __name__ == "__main__":
    main()
