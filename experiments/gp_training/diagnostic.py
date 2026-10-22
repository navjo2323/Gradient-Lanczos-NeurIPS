#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import tempfile
import pathlib
import sys
import time
from typing import Any

import numpy as np
import jax
import torch

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "lanczos_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

GAUSSIAN_PROCESS_DIR = pathlib.Path(__file__).resolve().parent
if str(GAUSSIAN_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_PROCESS_DIR))

from common import (  # noqa: E402
    Config,
    _reset_gpytorch_probe_cache,
    _torch_dtype_from_name,
    gpytorch,
    make_regression_data,
)
from train import (  # noqa: E402
    _build_model,
    _gpytorch_settings,
    _initialize_model,
    _load_initial_state,
    _make_probes,
    _snapshot_hypers,
    _sync,
    _torch_native_logdet_value_and_grad,
)


class _GpytorchProbeCapture:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}
        self._orig_forward = None

    def __enter__(self) -> "_GpytorchProbeCapture":
        from linear_operator.functions import _inv_quad_logdet

        self._orig_forward = _inv_quad_logdet.InvQuadLogdet.forward
        payload = self.payload
        orig_forward = self._orig_forward

        def forward_wrap(ctx, *args):
            out = orig_forward(ctx, *args)
            payload["probe_vectors"] = ctx.probe_vectors.detach().clone()
            payload["probe_vector_norms"] = ctx.probe_vector_norms.detach().clone()
            payload["num_random_probes"] = int(ctx.num_random_probes)
            return out

        _inv_quad_logdet.InvQuadLogdet.forward = staticmethod(forward_wrap)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        from linear_operator.functions import _inv_quad_logdet

        if self._orig_forward is not None:
            _inv_quad_logdet.InvQuadLogdet.forward = self._orig_forward


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _named_unique_trainable_params(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    named_params: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for prefix, module in (("model", model), ("likelihood", likelihood)):
        for name, param in module.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            named_params.append((f"{prefix}.{name}", param))
    return named_params


def _grads_to_vector(
    grads: list[torch.Tensor | None] | tuple[torch.Tensor | None, ...],
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).detach().reshape(-1))
        else:
            pieces.append(grad.detach().reshape(-1))
    return torch.cat(pieces)


def _rel_l2_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    numerator = torch.linalg.norm(approx - exact)
    denominator = torch.linalg.norm(exact).clamp_min(torch.finfo(exact.dtype).tiny)
    return float((numerator / denominator).detach().cpu().item())


def _dense_cholesky_logdet_and_grad(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    train_x: torch.Tensor,
    params: list[torch.nn.Parameter],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor | None], float]:
    _sync(device)
    t0 = time.perf_counter()
    output = likelihood(model(train_x))
    dense = output.lazy_covariance_matrix.to_dense()
    dense = 0.5 * (dense + dense.transpose(-1, -2))
    chol = torch.linalg.cholesky(dense)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum()
    grads = torch.autograd.grad(logdet, params, allow_unused=True)
    _sync(device)
    return logdet.detach(), list(grads), float(time.perf_counter() - t0)


def _gpytorch_logdet_and_grad(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    train_x: torch.Tensor,
    params: list[torch.nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
    probes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor | None], float, dict[str, Any]]:
    _reset_gpytorch_probe_cache()
    torch.manual_seed(int(args.probe_seed) + int(args.iteration))
    _sync(device)
    t0 = time.perf_counter()
    with _GpytorchProbeCapture() as capture, _gpytorch_settings(args):
        output = likelihood(model(train_x))
        covar = output.lazy_covariance_matrix
        if probes is not None:
            probe_vectors = probes.transpose(0, 1).contiguous()
            probe_norms = torch.linalg.norm(probe_vectors, dim=-2, keepdim=True).clamp_min(
                torch.finfo(probe_vectors.dtype).tiny
            )
            normalized_probe_vectors = probe_vectors / probe_norms
            covar._probe_vectors_and_norms = lambda: (normalized_probe_vectors, probe_norms)
        logdet = gpytorch.logdet(covar)
        grads = torch.autograd.grad(logdet, params, allow_unused=True)
    _sync(device)
    return logdet.detach(), list(grads), float(time.perf_counter() - t0), capture.payload


def _ours_no_reorth_logdet_and_grad(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    train_x: torch.Tensor,
    params: list[torch.nn.Parameter],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    probes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor | None], dict[str, float]]:
    if probes is None:
        probes = _make_probes(
            train_x.shape[0],
            args.probes,
            dtype,
            device,
            seed=int(args.probe_seed) + int(args.iteration),
        )
    with _gpytorch_settings(args):
        logdet, timings, grads = _torch_native_logdet_value_and_grad(
            model,
            likelihood,
            train_x,
            probes,
            args,
            "no_reorth",
            params,
            device,
        )
    return logdet.detach(), grads, timings


def _fresh_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    initial_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, list[tuple[str, torch.nn.Parameter]]]:
    model, likelihood = _build_model(
        train_x,
        train_y,
        ard=not args.no_ard,
        kernel_backend=args.kernel_backend,
        noise_min=args.noise_min,
        dtype=dtype,
        device=device,
    )
    _load_initial_state(model, initial_state)
    model.train()
    likelihood.train()
    named_params = _named_unique_trainable_params(model, likelihood)
    return model, likelihood, named_params


def _per_parameter_errors(
    names: list[str],
    params: list[torch.nn.Parameter],
    approx_grads: list[torch.Tensor | None],
    exact_grads: list[torch.Tensor | None],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name, param, approx, exact in zip(names, params, approx_grads, exact_grads):
        approx_t = torch.zeros_like(param).detach() if approx is None else approx.detach()
        exact_t = torch.zeros_like(param).detach() if exact is None else exact.detach()
        denom = torch.linalg.norm(exact_t.reshape(-1)).clamp_min(torch.finfo(param.dtype).tiny)
        errors[name] = float((torch.linalg.norm((approx_t - exact_t).reshape(-1)) / denom).detach().cpu().item())
    return errors


def _plot(payload: dict[str, Any], out_png: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    methods = ["gpytorch_preconditioned", "ours_no_reorth_preconditioned"]
    labels = ["GPyTorch", "Ours no reorth."]
    colors = ["#2ca02c", "#1f77b4"]
    logdet_errors = [max(float(payload["methods"][m]["logdet_rel_error_to_dense"]), 1e-16) for m in methods]
    grad_errors = [max(float(payload["methods"][m]["grad_rel_error_to_dense"]), 1e-16) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
    for ax, values, ylabel in (
        (axes[0], logdet_errors, "Relative logdet error"),
        (axes[1], grad_errors, "Relative gradient error"),
    ):
        ax.bar(labels, values, color=colors, width=0.58)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if gpytorch is None:
        raise ImportError("GPyTorch is not available in this environment.")

    jax.config.update("jax_enable_x64", True)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = _device_from_arg(args.device)
    dtype = _torch_dtype_from_name(args.torch_dtype)
    data_config = Config(
        n=args.n,
        seed=args.seed,
        dataset=args.dataset,
        all_data=args.n <= 0,
        kernel="matern32",
        torch_dtype=args.torch_dtype,
    )
    x_jax, y_jax, metadata = make_regression_data(data_config)
    x_np = np.asarray(x_jax)
    y_np = np.asarray(y_jax)

    split_gen = torch.Generator()
    split_gen.manual_seed(int(args.seed))
    perm = torch.randperm(x_np.shape[0], generator=split_gen)
    n_train = int(args.train_frac * x_np.shape[0])
    train_idx = perm[:n_train].numpy()
    train_x = torch.tensor(x_np[train_idx], dtype=dtype, device=device).contiguous()
    train_y = torch.tensor(y_np[train_idx], dtype=dtype, device=device).contiguous()

    initial_model, initial_likelihood = _build_model(
        train_x,
        train_y,
        ard=not args.no_ard,
        kernel_backend=args.kernel_backend,
        noise_min=args.noise_min,
        dtype=dtype,
        device=device,
    )
    initial_state = _initialize_model(initial_model, initial_likelihood, args.seed)
    initial_hypers = _snapshot_hypers(initial_model, initial_likelihood)
    del initial_model, initial_likelihood

    model, likelihood, named_params = _fresh_model(train_x, train_y, initial_state, args, dtype, device)
    names = [name for name, _param in named_params]
    params = [param for _name, param in named_params]
    print("[dense] exact Cholesky logdet gradient", flush=True)
    dense_logdet, dense_grads, dense_sec = _dense_cholesky_logdet_and_grad(model, likelihood, train_x, params, device)
    dense_vec = _grads_to_vector(dense_grads, params)

    shared_probes = _make_probes(
        train_x.shape[0],
        args.probes,
        dtype,
        device,
        seed=int(args.probe_seed) + int(args.iteration),
    )

    model, likelihood, named_params = _fresh_model(train_x, train_y, initial_state, args, dtype, device)
    params = [param for _name, param in named_params]
    print("[gpytorch] preconditioned SLQ logdet gradient", flush=True)
    gp_logdet, gp_grads, gp_sec, gp_capture = _gpytorch_logdet_and_grad(
        model,
        likelihood,
        train_x,
        params,
        args,
        device,
        probes=shared_probes,
    )
    gp_vec = _grads_to_vector(gp_grads, params)

    gp_probe_vectors = gp_capture["probe_vectors"].to(dtype=dtype, device=device)
    gp_probe_norms = gp_capture["probe_vector_norms"].to(dtype=dtype, device=device)
    num_random_probes = int(gp_capture["num_random_probes"])

    model, likelihood, named_params = _fresh_model(train_x, train_y, initial_state, args, dtype, device)
    params = [param for _name, param in named_params]
    print("[ours] preconditioned no-reorth Lanczos logdet gradient", flush=True)
    ours_logdet, ours_grads, ours_timings = _ours_no_reorth_logdet_and_grad(
        model,
        likelihood,
        train_x,
        params,
        args,
        dtype,
        device,
        probes=shared_probes,
    )
    ours_vec = _grads_to_vector(ours_grads, params)

    dense_value = float(dense_logdet.detach().cpu().item())
    methods = {
        "dense_cholesky": {
            "logdet": dense_value,
            "runtime_sec": dense_sec,
        },
        "gpytorch_preconditioned": {
            "logdet": float(gp_logdet.detach().cpu().item()),
            "runtime_sec": gp_sec,
            "logdet_rel_error_to_dense": abs(float(gp_logdet.detach().cpu().item()) - dense_value)
            / max(1.0, abs(dense_value)),
            "grad_rel_error_to_dense": _rel_l2_error(gp_vec, dense_vec),
            "per_parameter_grad_rel_error": _per_parameter_errors(names, params, gp_grads, dense_grads),
        },
        "ours_no_reorth_preconditioned": {
            "logdet": float(ours_logdet.detach().cpu().item()),
            "runtime_sec": float(
                ours_timings["lanczos_sec"] + ours_timings["slq_postprocess_sec"] + ours_timings["logdet_grad_sec"]
            ),
            "timings": ours_timings,
            "logdet_rel_error_to_dense": abs(float(ours_logdet.detach().cpu().item()) - dense_value)
            / max(1.0, abs(dense_value)),
            "grad_rel_error_to_dense": _rel_l2_error(ours_vec, dense_vec),
            "per_parameter_grad_rel_error": _per_parameter_errors(names, params, ours_grads, dense_grads),
        },
    }

    payload = {
        "script": pathlib.Path(__file__).name,
        "config": {
            **vars(args),
            "outdir": str(args.outdir),
            "device_resolved": str(device),
            "train_n": int(train_x.shape[0]),
            "dtype": str(dtype),
        },
        "dataset_metadata": metadata,
        "initial_hypers": initial_hypers,
        "parameter_names": names,
        "probe_summary": {
            "source": "shared_gaussian_passed_to_ours_and_gpytorch",
            "probe_vectors_shape": list(gp_capture["probe_vectors"].shape),
            "num_random_probes": num_random_probes,
            "probe_norm_mean": float(gp_probe_norms.mean().detach().cpu().item()),
            "probe_norm_min": float(gp_probe_norms.min().detach().cpu().item()),
            "probe_norm_max": float(gp_probe_norms.max().detach().cpu().item()),
        },
        "methods": methods,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.outdir / f"{args.name}_{metadata['dataset']}_n{x_np.shape[0]}_m{args.depth}_p{args.probes}_seed{args.seed}"
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_path = out_prefix.with_name(out_prefix.name + "_errors.png")
    _plot(payload, plot_path)
    payload["plot_path"] = str(plot_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-iteration logdet gradient diagnostic for GP training.")
    parser.add_argument("--dataset", default="protein")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--kernel-backend", choices=["standard", "keops"], default="standard")
    parser.add_argument("--no-ard", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--rank-precon", type=int, default=15)
    parser.add_argument(
        "--precondition-logdet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run our Lanczos diagnostic on P^{-1/2} K P^{-1/2}.",
    )
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--eval-cg-tol", type=float, default=1e-4)
    parser.add_argument("--noise-min", type=float, default=1e-4)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--deterministic-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("outputs/gp_training"))
    parser.add_argument("--name", default="protein_first_iter_precon15_m40_p20")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    for method_name, row in payload["methods"].items():
        if method_name == "dense_cholesky":
            continue
        print(
            f"{method_name}: "
            f"logdet_rel={row['logdet_rel_error_to_dense']:.3e}, "
            f"grad_rel={row['grad_rel_error_to_dense']:.3e}"
        )


if __name__ == "__main__":
    main()
