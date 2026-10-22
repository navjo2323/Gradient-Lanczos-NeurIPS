#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
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
    DiagLinearOperator,
    KeOpsMaternKernel,
    MaternKernel,
    ScaleKernel,
    _reset_gpytorch_probe_cache,
    _torch_dtype_from_name,
    gpytorch,
    lanczos_tridiag,
    make_regression_data,
)


@dataclasses.dataclass
class IterationRecord:
    method: str
    iteration: int
    loss: float
    iteration_time_sec: float
    cumulative_train_time_sec: float
    inv_quad_backward_sec: float | None = None
    lanczos_sec: float | None = None
    slq_postprocess_sec: float | None = None
    logdet_grad_sec: float | None = None
    optimizer_step_sec: float | None = None
    ritz_min: float | None = None
    ritz_max: float | None = None
    rmse: float | None = None
    eval_time_sec: float | None = None
    lengthscale_mean: float | None = None
    outputscale: float | None = None
    noise: float | None = None
    mean_constant: float | None = None


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _unique_trainable_params(model: torch.nn.Module, likelihood: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in list(model.parameters()) + list(likelihood.parameters()):
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    return params


def _stable_log_divided_difference(evals: torch.Tensor) -> torch.Tensor:
    a = evals[:, :, None]
    b = evals[:, None, :]
    denom = a + b
    d = (a - b) / denom
    abs_d = d.abs()
    d2 = d * d
    ratio_series = 1.0 + d2 / 3.0 + (d2 * d2) / 5.0 + (d2 * d2 * d2) / 7.0
    d_safe = torch.where(abs_d < 1e-4, torch.ones_like(d), d)
    ratio_exact = torch.atanh(d_safe) / d_safe
    ratio = torch.where(abs_d < 1e-4, ratio_series, ratio_exact)
    return (2.0 / denom) * ratio


def _projected_logdet_and_g(t_mat: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(t_mat)
    c = evecs[:, 0, :]
    probe_weights = torch.full((t_mat.shape[0],), float(n), dtype=t_mat.dtype, device=t_mat.device)
    logdet = torch.mean(probe_weights * torch.sum(c.square() * torch.log(evals), dim=1))

    f = _stable_log_divided_difference(evals)
    mid = (c[:, :, None] * c[:, None, :]) * f
    g_mat = probe_weights[:, None, None] * torch.einsum("pim,pmk,pjk->pij", evecs, mid, evecs)
    g_mat = 0.5 * (g_mat + g_mat.transpose(-1, -2))
    return logdet, g_mat, evals


def _lanczos_qt_no_reorth_matmul(
    matmul_closure: Any,
    probes: torch.Tensor,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_count, n = probes.shape
    dtype = probes.dtype
    device = probes.device
    init_vecs = probes.T.contiguous()
    q_mat = torch.empty(p_count, n, m, dtype=dtype, device=device)
    t_mat = torch.zeros(p_count, m, m, dtype=dtype, device=device)

    q_prev = torch.zeros_like(init_vecs)
    beta_prev = torch.zeros(p_count, dtype=dtype, device=device)
    q = init_vecs / torch.linalg.norm(init_vecs, dim=0, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    for k in range(m):
        q_mat[:, :, k] = q.T
        residual = matmul_closure(q)
        if k > 0:
            residual = residual - q_prev * beta_prev.unsqueeze(0)
        alpha = torch.sum(q * residual, dim=0)
        residual = residual - q * alpha.unsqueeze(0)
        t_mat[:, k, k] = alpha
        if k + 1 < m:
            beta = torch.linalg.norm(residual, dim=0)
            t_mat[:, k, k + 1] = beta
            t_mat[:, k + 1, k] = beta
            q_prev = q
            q = residual / beta.clamp_min(torch.finfo(dtype).tiny).unsqueeze(0)
            beta_prev = beta

    return q_mat, t_mat


def _lanczos_qt_no_reorth(covar: Any, probes: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _lanczos_qt_no_reorth_matmul(lambda rhs: covar.matmul(rhs), probes, m)


def _lanczos_qt_full_reorth_matmul(
    matmul_closure: Any,
    probes: torch.Tensor,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lanczos_tridiag is None:
        raise ImportError("linear_operator Lanczos is unavailable")
    p_count, n = probes.shape
    q_mat, t_mat = lanczos_tridiag(
        matmul_closure=matmul_closure,
        max_iter=m,
        dtype=probes.dtype,
        device=probes.device,
        matrix_shape=torch.Size([n, n]),
        batch_shape=torch.Size([]),
        init_vecs=probes.T.contiguous(),
    )
    if q_mat.ndim == 2:
        q_mat = q_mat.unsqueeze(0)
    if t_mat.ndim == 2:
        t_mat = t_mat.unsqueeze(0)
    return q_mat[:p_count], t_mat[:p_count]


def _lanczos_qt_full_reorth(covar: Any, probes: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _lanczos_qt_full_reorth_matmul(lambda rhs: covar.matmul(rhs), probes, m)


def _make_probes(n: int, p_count: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    base_samples = torch.randn(n, p_count, dtype=dtype, device=device, generator=generator)
    return base_samples.T.contiguous()


def _extract_low_rank_plus_diag_preconditioner(precond_lt: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if precond_lt is None or not hasattr(precond_lt, "linear_ops"):
        raise RuntimeError("Expected a low-rank plus diagonal pivoted-Cholesky preconditioner.")

    root = None
    diag = None
    for linear_op in precond_lt.linear_ops:
        name = linear_op.__class__.__name__
        if "RootLinearOperator" in name:
            root = linear_op.representation()[0]
        elif "DiagLinearOperator" in name:
            diag = linear_op.representation()[0]

    if root is None or diag is None:
        raise RuntimeError(f"Could not extract root/diag from preconditioner {type(precond_lt)}.")
    return root, diag


def _constant_diag_scalar(diag: torch.Tensor) -> torch.Tensor:
    diag_flat = diag.reshape(-1)
    if diag_flat.numel() == 1:
        return diag_flat[0]
    first = diag_flat[0]
    tol = 64.0 * torch.finfo(diag.dtype).eps * first.detach().abs().clamp_min(1.0)
    if not torch.allclose(diag_flat, first.expand_as(diag_flat), rtol=0.0, atol=float(tol.detach().cpu().item())):
        raise RuntimeError("Only scalar/constant diagonal pivoted-Cholesky preconditioners are supported.")
    return first


def _low_rank_plus_diag_inv_sqrt_apply(root: torch.Tensor, diag: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply (d I + R R^T)^(-1/2) to rhs without materializing the n x n matrix."""
    d = _constant_diag_scalar(diag)
    d_inv_sqrt = torch.rsqrt(d)
    gram = root.transpose(-2, -1).matmul(root)
    evals, evecs = torch.linalg.eigh(gram)
    eps = torch.finfo(root.dtype).eps
    eval_floor = eps * max(root.shape[0], 1)
    safe_evals = evals.clamp_min(eval_floor)
    delta = torch.rsqrt(d + safe_evals) - d_inv_sqrt
    limit = -0.5 * d.pow(-1.5)
    coeff_diag = torch.where(evals > eval_floor, delta / safe_evals, limit.expand_as(evals))
    coeff = evecs.matmul(torch.diag_embed(coeff_diag)).matmul(evecs.transpose(-2, -1))
    return d_inv_sqrt * rhs + root.matmul(coeff.matmul(root.transpose(-2, -1).matmul(rhs)))


def _snapshot_hypers(model: torch.nn.Module, likelihood: torch.nn.Module) -> dict[str, float]:
    lengthscale = model.covar_module.base_kernel.lengthscale.detach()
    return {
        "lengthscale_mean": float(lengthscale.mean().cpu().item()),
        "outputscale": float(model.covar_module.outputscale.detach().cpu().item()),
        "noise": float(likelihood.noise.detach().mean().cpu().item()),
        "mean_constant": float(model.mean_module.constant.detach().cpu().item()),
    }


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: torch.nn.Module, *, ard: bool, kernel_backend: str):
        super().__init__(train_x, train_y, likelihood)
        x_dim = train_x.shape[-1]
        ard_num_dims = x_dim if ard else None
        if kernel_backend == "keops":
            if KeOpsMaternKernel is None:
                raise ImportError("Requested KeOps kernel, but GPyTorch KeOps MaternKernel is unavailable")
            base_kernel = KeOpsMaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        elif kernel_backend == "standard":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Unknown kernel backend: {kernel_backend}")
        self.covar_module = ScaleKernel(base_kernel)
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _build_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    ard: bool,
    kernel_backend: str,
    noise_min: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[ExactGPModel, torch.nn.Module]:
    positive = gpytorch.constraints.GreaterThan(noise_min)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=positive).to(dtype=dtype, device=device)
    model = ExactGPModel(train_x, train_y, likelihood, ard=ard, kernel_backend=kernel_backend).to(dtype=dtype, device=device)
    return model, likelihood


def _initialize_model(model: ExactGPModel, likelihood: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    hypers = {
        "likelihood.noise_covar.raw_noise": torch.randn_like(likelihood.noise_covar.raw_noise),
        "covar_module.base_kernel.raw_lengthscale": torch.randn_like(model.covar_module.base_kernel.raw_lengthscale),
        "covar_module.raw_outputscale": torch.randn_like(model.covar_module.raw_outputscale),
        "mean_module.raw_constant": torch.randn_like(model.mean_module.raw_constant),
    }
    model.initialize(**hypers)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _load_initial_state(model: ExactGPModel, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.detach().clone() for key, value in state.items()})


@contextlib.contextmanager
def _gpytorch_settings(args: argparse.Namespace):
    with (
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.max_preconditioner_size(args.rank_precon),
        gpytorch.settings.min_preconditioning_size(10),
        gpytorch.settings.cg_tolerance(args.cg_tol),
        gpytorch.settings.num_trace_samples(args.probes),
        gpytorch.settings.max_lanczos_quadrature_iterations(args.depth),
        gpytorch.settings.max_root_decomposition_size(args.depth),
        gpytorch.settings.fast_computations(True, True, True),
        gpytorch.settings.deterministic_probes(bool(args.deterministic_probes)),
        gpytorch.settings.tridiagonal_jitter(0.0),
    ):
        yield


def _evaluate_rmse(
    model: ExactGPModel,
    likelihood: torch.nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    likelihood.eval()
    _sync(device)
    t0 = time.perf_counter()
    with (
        torch.no_grad(),
        _gpytorch_settings(args),
        gpytorch.settings.cg_tolerance(args.eval_cg_tol),
        gpytorch.settings.skip_posterior_variances(),
    ):
        pred = likelihood(model(test_x))
        rmse = torch.sqrt(torch.mean((pred.mean - test_y) ** 2))
    _sync(device)
    elapsed = time.perf_counter() - t0
    if was_training:
        model.train()
        likelihood.train()
    return float(rmse.detach().cpu().item()), float(elapsed)


def _torch_native_logdet_value_and_grad(
    model: ExactGPModel,
    likelihood: torch.nn.Module,
    train_x: torch.Tensor,
    probes: torch.Tensor,
    args: argparse.Namespace,
    forward_impl: str,
    params: list[torch.nn.Parameter],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], list[torch.Tensor | None]]:
    n = train_x.shape[0]
    use_preconditioned_logdet = bool(args.precondition_logdet and args.rank_precon > 0)

    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ng = likelihood(model(train_x))
        covar_ng = output_ng.lazy_covariance_matrix
        logdet_p_ng = None
        if use_preconditioned_logdet:
            _preconditioner_ng, precond_lt_ng, logdet_p_ng = covar_ng._preconditioner()
            root_ng, diag_ng = _extract_low_rank_plus_diag_preconditioner(precond_lt_ng)

            def matmul_preconditioned_ng(rhs: torch.Tensor) -> torch.Tensor:
                right = _low_rank_plus_diag_inv_sqrt_apply(root_ng, diag_ng, rhs)
                return _low_rank_plus_diag_inv_sqrt_apply(root_ng, diag_ng, covar_ng.matmul(right))

            matmul_closure = matmul_preconditioned_ng
        else:
            matmul_closure = lambda rhs: covar_ng.matmul(rhs)

        if forward_impl == "no_reorth":
            q_mat, t_mat = _lanczos_qt_no_reorth_matmul(matmul_closure, probes, args.depth)
        elif forward_impl == "full_reorth":
            q_mat, t_mat = _lanczos_qt_full_reorth_matmul(matmul_closure, probes, args.depth)
        else:
            raise ValueError(f"Unknown forward_impl: {forward_impl}")
    _sync(device)
    lanczos_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    slq_logdet, g_mat, evals = _projected_logdet_and_g(t_mat, n=n)
    logdet = slq_logdet if logdet_p_ng is None else slq_logdet + logdet_p_ng
    w_mat = torch.einsum("pnm,pmk->pnk", q_mat, g_mat).detach()
    q_mat = q_mat.detach()
    _sync(device)
    post_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    output_grad = likelihood(model(train_x))
    covar_grad = output_grad.lazy_covariance_matrix
    p_count, _, depth = q_mat.shape
    v_block = q_mat.permute(1, 0, 2).reshape(n, p_count * depth)
    w_block = w_mat.permute(1, 0, 2).reshape(n, p_count * depth)
    if use_preconditioned_logdet:
        _preconditioner_grad, precond_lt_grad, logdet_p_grad = covar_grad._preconditioner()
        root_grad, diag_grad = _extract_low_rank_plus_diag_preconditioner(precond_lt_grad)
        right = _low_rank_plus_diag_inv_sqrt_apply(root_grad, diag_grad, v_block)
        av_block = _low_rank_plus_diag_inv_sqrt_apply(root_grad, diag_grad, covar_grad.matmul(right))
        contraction = logdet_p_grad + torch.sum(w_block * av_block) / float(p_count)
    else:
        kv_block = covar_grad.matmul(v_block)
        contraction = torch.sum(w_block * kv_block) / float(p_count)
    grads = torch.autograd.grad(contraction, params, allow_unused=True)
    _sync(device)
    grad_sec = time.perf_counter() - t2

    timings = {
        "lanczos_sec": float(lanczos_sec),
        "slq_postprocess_sec": float(post_sec),
        "logdet_grad_sec": float(grad_sec),
        "logdet_preconditioned": bool(use_preconditioned_logdet),
        "ritz_min": float(evals.detach().min().cpu().item()),
        "ritz_max": float(evals.detach().max().cpu().item()),
    }
    return logdet.detach(), timings, list(grads)


def _train_one_method(
    method: str,
    initial_state: dict[str, torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    if method == "ours_no_reorth":
        forward_impl = "no_reorth"
    elif method == "ours_full_reorth":
        forward_impl = "full_reorth"
    elif method == "gpytorch":
        forward_impl = ""
    else:
        raise ValueError(f"Unknown method: {method}")

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
    params = _unique_trainable_params(model, likelihood)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    records: list[IterationRecord] = []

    cumulative = 0.0
    for iteration in range(1, args.epochs + 1):
        _sync(device)
        t_iter = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        if method == "gpytorch":
            if args.deterministic_probes:
                _reset_gpytorch_probe_cache()
                torch.manual_seed(args.probe_seed + iteration)
            with _gpytorch_settings(args):
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
            component_times: dict[str, float | None] = {
                "inv_quad_backward_sec": None,
                "lanczos_sec": None,
                "slq_postprocess_sec": None,
                "logdet_grad_sec": None,
                "ritz_min": None,
                "ritz_max": None,
            }
        else:
            with _gpytorch_settings(args):
                t_inv = time.perf_counter()
                output = likelihood(model(train_x))
                centered = train_y - output.mean
                inv_quad = output.lazy_covariance_matrix.inv_quad(centered.unsqueeze(-1))
                inv_loss = 0.5 * inv_quad / float(train_x.shape[0])
                inv_loss.backward()
                _sync(device)
                inv_quad_backward_sec = time.perf_counter() - t_inv

                probes = _make_probes(
                    train_x.shape[0],
                    args.probes,
                    dtype,
                    device,
                    seed=args.probe_seed + iteration,
                )
                logdet, timings, logdet_grads = _torch_native_logdet_value_and_grad(
                    model,
                    likelihood,
                    train_x,
                    probes,
                    args,
                    forward_impl,
                    params,
                    device,
                )
                scale = 0.5 / float(train_x.shape[0])
                for param, grad in zip(params, logdet_grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = scale * grad.detach()
                    else:
                        param.grad.add_(scale * grad.detach())

                constant = 0.5 * math.log(2.0 * math.pi)
                loss = inv_loss.detach() + 0.5 * logdet / float(train_x.shape[0]) + constant
                component_times = {
                    "inv_quad_backward_sec": float(inv_quad_backward_sec),
                    "lanczos_sec": timings["lanczos_sec"],
                    "slq_postprocess_sec": timings["slq_postprocess_sec"],
                    "logdet_grad_sec": timings["logdet_grad_sec"],
                    "ritz_min": timings["ritz_min"],
                    "ritz_max": timings["ritz_max"],
                }

        if args.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        t_step = time.perf_counter()
        optimizer.step()
        _sync(device)
        optimizer_step_sec = time.perf_counter() - t_step
        iter_time = time.perf_counter() - t_iter
        cumulative += iter_time

        rmse = None
        eval_time = None
        if args.eval_every > 0 and (iteration % args.eval_every == 0 or iteration == args.epochs):
            rmse, eval_time = _evaluate_rmse(model, likelihood, test_x, test_y, args, device)

        hypers = _snapshot_hypers(model, likelihood)
        record = IterationRecord(
            method=method,
            iteration=iteration,
            loss=float(loss.detach().cpu().item()),
            iteration_time_sec=float(iter_time),
            cumulative_train_time_sec=float(cumulative),
            inv_quad_backward_sec=component_times["inv_quad_backward_sec"],
            lanczos_sec=component_times["lanczos_sec"],
            slq_postprocess_sec=component_times["slq_postprocess_sec"],
            logdet_grad_sec=component_times["logdet_grad_sec"],
            optimizer_step_sec=float(optimizer_step_sec),
            ritz_min=component_times["ritz_min"],
            ritz_max=component_times["ritz_max"],
            rmse=rmse,
            eval_time_sec=eval_time,
            **hypers,
        )
        records.append(record)
        print(
            f"{method:>18} iter={iteration:03d} loss={record.loss:.5f} "
            f"time={record.iteration_time_sec:.3f}s rmse={record.rmse}",
            flush=True,
        )

    final_rmse, final_eval_time = _evaluate_rmse(model, likelihood, test_x, test_y, args, device)
    final_hypers = _snapshot_hypers(model, likelihood)
    return {
        "method": method,
        "final_loss": records[-1].loss if records else float("nan"),
        "final_rmse": final_rmse,
        "final_eval_time_sec": final_eval_time,
        "train_time_sec": float(sum(record.iteration_time_sec for record in records)),
        "mean_iteration_time_sec": float(np.mean([record.iteration_time_sec for record in records])),
        "best_iteration_time_sec": float(np.min([record.iteration_time_sec for record in records])),
        "final_hypers": final_hypers,
        "history": [dataclasses.asdict(record) for record in records],
    }


def _plot_results(result: dict[str, Any], out_prefix: pathlib.Path) -> list[str]:
    import matplotlib.pyplot as plt

    styles = {
        "gpytorch": {"label": "GPyTorch", "color": "C2", "linestyle": "--"},
        "ours_no_reorth": {"label": "Ours no reorth.", "color": "C0", "linestyle": "-"},
        "ours_full_reorth": {"label": "Ours full reorth.", "color": "C3", "linestyle": ":"},
    }
    saved: list[str] = []

    fig, (ax_loss, ax_time) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for method_result in result["methods"]:
        method = method_result["method"]
        style = styles.get(method, {"label": method, "color": None, "linestyle": "-"})
        hist = method_result["history"]
        x = [row["iteration"] for row in hist]
        ax_loss.plot(
            x,
            [row["loss"] for row in hist],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
        ax_time.plot(
            x,
            [row["iteration_time_sec"] for row in hist],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Train negative log likelihood / n")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)
    ax_time.set_xlabel("Iteration")
    ax_time.set_ylabel("Iteration time (s)")
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(fontsize=8)
    fig.tight_layout()
    path = out_prefix.with_name(out_prefix.name + "_loss_time.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plotted_rmse = False
    for method_result in result["methods"]:
        method = method_result["method"]
        style = styles.get(method, {"label": method, "color": None, "linestyle": "-"})
        hist = [row for row in method_result["history"] if row["rmse"] is not None]
        if hist:
            x_values = [row["iteration"] for row in hist]
            y_values = [row["rmse"] for row in hist]
        else:
            x_values = [len(method_result["history"])]
            y_values = [method_result["final_rmse"]]
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker="o",
            markersize=4.0,
        )
        plotted_rmse = True
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test RMSE")
    ax.grid(True, alpha=0.3)
    if plotted_rmse:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_prefix.with_name(out_prefix.name + "_rmse.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def _parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"gpytorch", "ours_no_reorth", "ours_full_reorth"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; expected subset of {sorted(allowed)}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exact GPs with GPyTorch vs Torch-native Lanczos logdet gradients.")
    parser.add_argument("--dataset", default="protein")
    parser.add_argument("--n", type=int, default=8192, help="Total selected data before train/test split; <=0 means all data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--kernel-backend", choices=["standard", "keops"], default="standard")
    parser.add_argument("--no-ard", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--rank-precon", type=int, default=10)
    parser.add_argument(
        "--precondition-logdet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For our methods, run Lanczos on P^{-1/2} K P^{-1/2} and differentiate through the pivoted-Cholesky preconditioner.",
    )
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--eval-cg-tol", type=float, default=1e-4)
    parser.add_argument("--noise-min", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=0, help="0 evaluates only after training.")
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--deterministic-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--methods", default="gpytorch,ours_no_reorth,ours_full_reorth")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("outputs/gp_training"))
    parser.add_argument("--name", default="torch_native_train")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = _device_from_arg(args.device)
    dtype = _torch_dtype_from_name(args.torch_dtype)
    methods = _parse_methods(args.methods)

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
    test_idx = perm[n_train:].numpy()
    train_x = torch.tensor(x_np[train_idx], dtype=dtype, device=device).contiguous()
    train_y = torch.tensor(y_np[train_idx], dtype=dtype, device=device).contiguous()
    test_x = torch.tensor(x_np[test_idx], dtype=dtype, device=device).contiguous()
    test_y = torch.tensor(y_np[test_idx], dtype=dtype, device=device).contiguous()

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
    del initial_model, initial_likelihood

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.outdir / f"{args.name}_{metadata['dataset']}_n{x_np.shape[0]}_m{args.depth}_p{args.probes}_seed{args.seed}"

    results = []
    total_t0 = time.perf_counter()
    for method in methods:
        print(f"=== Training {method} ===", flush=True)
        results.append(
            _train_one_method(
                method,
                initial_state,
                train_x,
                train_y,
                test_x,
                test_y,
                args,
                dtype,
                device,
            )
        )

    result = {
        "script": pathlib.Path(__file__).name,
        "config": {
            **vars(args),
            "outdir": str(args.outdir),
            "device_resolved": str(device),
            "train_n": int(train_x.shape[0]),
            "test_n": int(test_x.shape[0]),
            "dtype": str(dtype),
        },
        "dataset_metadata": metadata,
        "methods": results,
        "total_wall_time_sec": float(time.perf_counter() - total_t0),
    }
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    if not args.no_plots:
        saved = _plot_results(result, out_prefix)
        print(json.dumps({"saved_plots": saved}, indent=2))


if __name__ == "__main__":
    main()
