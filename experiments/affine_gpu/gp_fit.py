#!/usr/bin/env python3
"""End-to-end matrix-free Hadamard Matérn GP hyperparameter fit.

The covariance is K = A^(-alpha), A = mass I + scale B. Synthetic samples are
drawn exactly in the hidden Hadamard spectral basis, but compared methods see
only matrix-free matvecs with A.

The NLL is

    0.5 * sum_i y_i^T A^alpha y_i - 0.5 * samples * alpha * logdet(A) + prior.

The logdet term is evaluated exactly for every method so the experiment
isolates the data-quadratic derivative. In larger graph settings this term can
be replaced by SLQ/GPyTorch without changing the projected data-gradient rule.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .optimizer import (
    adam_update,
    timed,
)
from .kernels import (
    Operator,
    chebyshev_adjoint_affine,
    choose_device,
    fwht,
    lanczos,
)


@dataclass(frozen=True)
class FitRow:
    method: str
    iteration: int
    seconds: float
    nll: float
    exact_nll_at_params: float
    data_value: float
    exact_data_value: float
    logdet_term: float
    log_mass: float
    log_scale: float
    alpha: float
    mass: float
    scale: float
    grad_mass: float
    grad_scale: float
    grad_alpha: float
    exact_grad_mass: float
    exact_grad_scale: float
    exact_grad_alpha: float
    rel_grad_vs_exact: float
    rel_data_value_vs_exact: float
    rel_param_error: float
    notes: str


def make_samples(
    op: Operator,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert op.cluster_spectrum is not None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.sample_seed)
    spectrum = op.cluster_spectrum.detach().cpu().to(dtype=torch.float64)
    true_eigs = args.true_mass + args.true_scale * spectrum
    std = torch.pow(true_eigs, -0.5 * args.true_alpha)
    white = torch.randn((args.samples, args.n), dtype=torch.float64, generator=generator)
    spectral_samples = white * std[None, :]
    if args.operator == "hadamard-clusters":
        samples = torch.stack([fwht(sample.to(dtype=dtype)) / math.sqrt(args.n) for sample in spectral_samples], dim=0)
    else:
        samples = spectral_samples.to(dtype=dtype)
    return samples.to(device=device, dtype=dtype)


def logdet_and_gradient(op: Operator, params: np.ndarray, sample_count: int) -> tuple[float, np.ndarray]:
    assert op.cluster_spectrum is not None
    spectrum = op.cluster_spectrum.detach().cpu().to(dtype=torch.float64)
    mass = math.exp(float(params[0]))
    scale = math.exp(float(params[1]))
    alpha = float(params[2])
    eigenvalues = mass + scale * spectrum
    log_eigs = torch.log(eigenvalues)
    logdet = float(torch.sum(log_eigs))
    scale_factor = float(op.args.logdet_scale)
    # Term in NLL: -0.5 * logdet_scale * s * alpha * logdet(A)
    value = -0.5 * scale_factor * float(sample_count) * alpha * logdet
    gradient = np.asarray(
        [
            -0.5 * scale_factor * float(sample_count) * alpha * float(torch.sum(mass / eigenvalues)),
            -0.5 * scale_factor * float(sample_count) * alpha * float(torch.sum(scale * spectrum / eigenvalues)),
            -0.5 * scale_factor * float(sample_count) * logdet,
        ],
        dtype=np.float64,
    )
    return value, gradient


def parameter_prior(params: np.ndarray, center: np.ndarray, strength: float) -> tuple[float, np.ndarray]:
    delta = params - center
    return 0.5 * strength * float(delta @ delta), strength * delta


def bounds_for_params(op: Operator, params: np.ndarray) -> tuple[float, float]:
    assert op.cluster_spectrum is not None
    mass = math.exp(float(params[0]))
    scale = math.exp(float(params[1]))
    spectrum = op.cluster_spectrum.detach().cpu().to(dtype=torch.float64)
    lower = mass + scale * float(torch.min(spectrum))
    upper = mass + scale * float(torch.max(spectrum)) * (1.0 + 1.0e-6)
    return max(lower * (1.0 - 1.0e-8), 1.0e-30), upper


def positive_power_sensitivity(
    tri: torch.Tensor,
    input_norm: torch.Tensor,
    alpha: float,
    lower: float,
    upper: float,
    ritz_guard: str,
) -> tuple[float, torch.Tensor, float]:
    small = tri.detach().cpu().to(dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(small)
    if ritz_guard == "clip":
        eigenvalues = torch.clamp(eigenvalues, min=float(lower), max=float(upper))
    elif ritz_guard != "none":
        raise ValueError(f"unknown Ritz guard {ritz_guard!r}")
    if torch.any(eigenvalues <= 0.0):
        nan = torch.full_like(small, float("nan"))
        return float("nan"), nan.to(device=tri.device, dtype=tri.dtype), float("nan")
    values = torch.pow(eigenvalues, float(alpha))
    derivatives = float(alpha) * torch.pow(eigenvalues, float(alpha) - 1.0)
    left = eigenvalues[:, None]
    right = eigenvalues[None, :]
    denominator = left - right
    divided = (values[:, None] - values[None, :]) / denominator
    scale = torch.clamp(torch.max(torch.abs(eigenvalues)), min=1.0)
    close = torch.abs(denominator) <= 1.0e-8 * scale
    divided = torch.where(close, derivatives[:, None].expand_as(divided), divided)
    e1 = torch.zeros(eigenvalues.numel(), dtype=torch.float64)
    e1[0] = 1.0
    coeffs = eigenvectors.T @ e1
    input_scale = input_norm.detach().cpu().to(dtype=torch.float64)
    input_scale = input_scale * input_scale
    sensitivity = input_scale * eigenvectors @ ((coeffs[:, None] * coeffs[None, :]) * divided) @ eigenvectors.T
    value = float(input_scale * torch.sum(values * coeffs * coeffs))
    alpha_grad = float(input_scale * torch.sum(torch.log(eigenvalues) * values * coeffs * coeffs))
    return value, sensitivity.to(device=tri.device, dtype=tri.dtype), alpha_grad


def exact_positive_reference(
    op: Operator,
    initial: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    assert op.cluster_spectrum is not None
    vector = initial.detach().cpu().to(dtype=torch.float64)
    if args.operator == "hadamard-clusters":
        vector = fwht(vector) / math.sqrt(args.n)
    spectrum = op.cluster_spectrum.detach().cpu().to(dtype=torch.float64)
    weights = vector * vector
    mass = math.exp(float(params[0]))
    scale = math.exp(float(params[1]))
    alpha = float(params[2])
    eigenvalues = mass + scale * spectrum
    values = torch.pow(eigenvalues, alpha)
    derivatives = alpha * torch.pow(eigenvalues, alpha - 1.0)
    value = float(torch.sum(weights * values))
    gradient = np.asarray(
        [
            float(torch.sum(weights * derivatives * mass)),
            float(torch.sum(weights * derivatives * scale * spectrum)),
            float(torch.sum(weights * torch.log(eigenvalues) * values)),
        ],
        dtype=np.float64,
    )
    return value, gradient


def evaluate_lanczos_positive(
    op: Operator,
    initial: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
    *,
    full_reorth: bool,
) -> tuple[float, np.ndarray]:
    dtype = initial.dtype
    device = initial.device
    log_mass = torch.tensor(float(params[0]), dtype=dtype, device=device, requires_grad=True)
    log_scale = torch.tensor(float(params[1]), dtype=dtype, device=device, requires_grad=True)
    tri, _, _, _, _, _ = lanczos(
        op,
        initial,
        log_mass,
        log_scale,
        args.steps,
        full_reorth=full_reorth,
        full_reorth_passes=args.full_reorth_passes,
        beta_breakdown_tolerance=args.beta_breakdown_tolerance,
        store_basis=False,
        check_orthogonality=False,
    )
    lower, upper = bounds_for_params(op, params)
    value, g_tri, alpha_grad = positive_power_sensitivity(
        tri,
        torch.linalg.norm(initial),
        float(params[2]),
        lower,
        upper,
        args.ritz_guard,
    )
    torch.autograd.backward(tri, grad_tensors=g_tri)
    return value, np.asarray(
        [float(log_mass.grad.detach().cpu()), float(log_scale.grad.detach().cpu()), alpha_grad],
        dtype=np.float64,
    )


def evaluate_projected_positive(
    op: Operator,
    initial: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    dtype = initial.dtype
    device = initial.device
    log_mass = torch.tensor(float(params[0]), dtype=dtype, device=device)
    log_scale = torch.tensor(float(params[1]), dtype=dtype, device=device)
    with torch.no_grad():
        tri, _, _, _, _, _ = lanczos(
            op,
            initial,
            log_mass,
            log_scale,
            args.steps,
            full_reorth=False,
            beta_breakdown_tolerance=args.beta_breakdown_tolerance,
            store_basis=False,
            check_orthogonality=False,
        )
    lower, upper = bounds_for_params(op, params)
    value, g_tri, alpha_grad = positive_power_sensitivity(
        tri,
        torch.linalg.norm(initial).detach(),
        float(params[2]),
        lower,
        upper,
        args.ritz_guard,
    )
    mass = torch.exp(log_mass)
    eye = torch.eye(tri.shape[0], dtype=tri.dtype, device=tri.device)
    grad_mass = torch.sum(g_tri * (mass * eye))
    grad_scale = torch.sum(g_tri * (tri - mass * eye))
    return value, np.asarray(
        [float(grad_mass.detach().cpu()), float(grad_scale.detach().cpu()), alpha_grad],
        dtype=np.float64,
    )


def cheb_coefficients_power(degree: int, lower: float, upper: float, alpha: float, *, alpha_derivative: bool) -> torch.Tensor:
    node_count = max(2 * degree + 1, 64)
    j = np.arange(node_count, dtype=np.float64)
    angles = np.pi * (j + 0.5) / node_count
    nodes = np.cos(angles)
    x = 0.5 * (upper + lower) + 0.5 * (upper - lower) * nodes
    values = np.power(x, alpha)
    if alpha_derivative:
        values = values * np.log(x)
    coeffs = np.empty(degree + 1, dtype=np.float64)
    coeffs[0] = np.mean(values)
    for k in range(1, degree + 1):
        coeffs[k] = 2.0 * np.mean(values * np.cos(k * angles))
    return torch.tensor(coeffs, dtype=torch.float32)


def evaluate_chebyshev_positive(
    op: Operator,
    initial: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    dtype = initial.dtype
    device = initial.device
    log_mass = torch.tensor(float(params[0]), dtype=dtype, device=device)
    log_scale = torch.tensor(float(params[1]), dtype=dtype, device=device)
    lower, upper = bounds_for_params(op, params)
    coeffs = cheb_coefficients_power(args.chebyshev_degree, lower, upper, float(params[2]), alpha_derivative=False)
    alpha_coeffs = cheb_coefficients_power(args.chebyshev_degree, lower, upper, float(params[2]), alpha_derivative=True)
    value, grad2, _, _ = chebyshev_adjoint_affine(op, initial, log_mass, log_scale, args, coeffs, lower, upper)
    alpha_value, _, _, _ = chebyshev_adjoint_affine(op, initial, log_mass, log_scale, args, alpha_coeffs, lower, upper)
    return value, np.asarray([float(grad2[0].detach().cpu()), float(grad2[1].detach().cpu()), alpha_value], dtype=np.float64)


def fwht_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("fwht_columns expects an n-by-batch matrix")
    n, batch = matrix.shape
    y = matrix
    h = 1
    while h < n:
        y = y.reshape(-1, 2 * h, batch)
        left = y[:, :h, :]
        right = y[:, h:, :]
        y = torch.cat((left + right, left - right), dim=1)
        h *= 2
    return y.reshape(n, batch)


def base_matvec_matrix(op: Operator, matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 1:
        return op.base_matvec(matrix)
    if op.args.operator == "diagonal-clusters":
        assert op.cluster_spectrum is not None
        return op.cluster_spectrum[:, None] * matrix
    if op.args.operator == "hadamard-clusters":
        assert op.cluster_spectrum is not None
        transformed = fwht_columns(matrix) / math.sqrt(op.n)
        return fwht_columns(op.cluster_spectrum[:, None] * transformed) / math.sqrt(op.n)
    return torch.stack([op.base_matvec(matrix[:, column]) for column in range(matrix.shape[1])], dim=1)


def matvec_matrix(op: Operator, matrix: torch.Tensor, log_mass: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_mass) * matrix + torch.exp(log_scale) * base_matvec_matrix(op, matrix)


def evaluate_chebyshev_positive_batch(
    op: Operator,
    samples: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, np.ndarray, float]:
    start = time.perf_counter()
    dtype = samples.dtype
    log_mass = torch.tensor(float(params[0]), dtype=dtype, device=device)
    log_scale = torch.tensor(float(params[1]), dtype=dtype, device=device)
    lower, upper = bounds_for_params(op, params)
    coeffs = cheb_coefficients_power(args.chebyshev_degree, lower, upper, float(params[2]), alpha_derivative=False)
    alpha_coeffs = cheb_coefficients_power(args.chebyshev_degree, lower, upper, float(params[2]), alpha_derivative=True)

    vectors = samples.T.contiguous()
    coeffs = coeffs.to(device=device, dtype=dtype)
    alpha_coeffs = alpha_coeffs.to(device=device, dtype=dtype)
    degree = coeffs.numel() - 1
    mass = torch.exp(log_mass)
    scale_param = torch.exp(log_scale)
    interval_scale = 2.0 / (upper - lower)

    def scaled_apply(matrix: torch.Tensor) -> torch.Tensor:
        return (2.0 * matvec_matrix(op, matrix, log_mass, log_scale) - (upper + lower) * matrix) / (upper - lower)

    with torch.no_grad():
        states: list[torch.Tensor] = [vectors]
        dot0 = torch.sum(vectors * vectors)
        value = coeffs[0] * dot0
        alpha_value = alpha_coeffs[0] * dot0
        if degree >= 1:
            states.append(scaled_apply(vectors))
            dot1 = torch.sum(vectors * states[1])
            value = value + coeffs[1] * dot1
            alpha_value = alpha_value + alpha_coeffs[1] * dot1
            for order in range(1, degree):
                next_state = 2.0 * scaled_apply(states[order]) - states[order - 1]
                states.append(next_state)
                dot = torch.sum(vectors * next_state)
                value = value + coeffs[order + 1] * dot
                alpha_value = alpha_value + alpha_coeffs[order + 1] * dot

        gradient = torch.zeros(2, dtype=dtype, device=device)
        if degree >= 1:
            lambda_next2 = torch.zeros_like(vectors)
            lambda_next1 = coeffs[degree] * vectors
            for order in range(degree - 1, 0, -1):
                state = states[order]
                gradient[0] = gradient[0] + 2.0 * interval_scale * mass * torch.sum(lambda_next1 * state)
                gradient[1] = gradient[1] + 2.0 * interval_scale * scale_param * torch.sum(
                    lambda_next1 * base_matvec_matrix(op, state)
                )
                current = coeffs[order] * vectors + 2.0 * scaled_apply(lambda_next1) - lambda_next2
                lambda_next2 = lambda_next1
                lambda_next1 = current
            state0 = states[0]
            gradient[0] = gradient[0] + interval_scale * mass * torch.sum(lambda_next1 * state0)
            gradient[1] = gradient[1] + interval_scale * scale_param * torch.sum(lambda_next1 * base_matvec_matrix(op, state0))

    total_value = 0.5 * float(value.detach().cpu())
    total_gradient = 0.5 * np.asarray(
        [float(gradient[0].detach().cpu()), float(gradient[1].detach().cpu()), float(alpha_value.detach().cpu())],
        dtype=np.float64,
    )
    return total_value, total_gradient, time.perf_counter() - start


def exact_data_for_samples(
    op: Operator,
    samples: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    total = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    sample_args = args
    for sample in samples:
        value, grad = exact_positive_reference(op, sample, params, sample_args)
        total += 0.5 * value
        gradient += 0.5 * grad
    return total, gradient


def approximate_data_for_samples(
    method: str,
    op: Operator,
    samples: torch.Tensor,
    params: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, np.ndarray, float]:
    start = time.perf_counter()
    total = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    if method == "chebyshev":
        return evaluate_chebyshev_positive_batch(op, samples, params, args, device)
    for sample in samples:
        if method == "full":
            _, result = timed(device, lambda sample=sample: evaluate_lanczos_positive(op, sample, params, args, full_reorth=True))
        elif method == "no":
            _, result = timed(device, lambda sample=sample: evaluate_lanczos_positive(op, sample, params, args, full_reorth=False))
        elif method == "projected":
            _, result = timed(device, lambda sample=sample: evaluate_projected_positive(op, sample, params, args))
        else:
            raise ValueError(f"unknown approximate method {method!r}")
        value, grad = result
        total += 0.5 * value
        gradient += 0.5 * grad
    return total, gradient, time.perf_counter() - start


def evaluate_method(
    method: str,
    op: Operator,
    samples: torch.Tensor,
    params: np.ndarray,
    center: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, float, float, np.ndarray, float, float]:
    exact_data, exact_data_grad = exact_data_for_samples(op, samples, params, args)
    logdet_value, logdet_grad = logdet_and_gradient(op, params, samples.shape[0])
    prior_value, prior_grad = parameter_prior(params, center, args.prior_strength)
    exact_nll = exact_data + logdet_value + prior_value
    exact_grad = exact_data_grad + logdet_grad + prior_grad

    if method == "exact":
        data_value = exact_data
        data_grad = exact_data_grad
        seconds = 0.0
    else:
        data_value, data_grad, seconds = approximate_data_for_samples(method, op, samples, params, args, device)
    nll = data_value + logdet_value + prior_value
    gradient = data_grad + logdet_grad + prior_grad
    rel_grad = float(np.linalg.norm(gradient - exact_grad) / max(np.linalg.norm(exact_grad), 1.0))
    rel_value = abs(data_value - exact_data) / max(abs(exact_data), 1.0)
    return nll, data_value, exact_data, logdet_value, gradient, seconds, exact_nll, exact_grad, rel_grad, rel_value


def run(args: argparse.Namespace) -> list[FitRow]:
    device = choose_device(args.device)
    if args.require_gpu and device.type not in {"cuda", "mps"}:
        raise SystemExit("--require-gpu was set, but no GPU backend is available.")
    dtype = torch.float32
    op = Operator(args, device, dtype)
    samples = make_samples(op, args, device, dtype)
    params0 = np.asarray([math.log(args.init_mass), math.log(args.init_scale), args.init_alpha], dtype=np.float64)
    true_params = np.asarray([math.log(args.true_mass), math.log(args.true_scale), args.true_alpha], dtype=np.float64)
    center = np.asarray([math.log(args.prior_mass), math.log(args.prior_scale), args.prior_alpha], dtype=np.float64)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows: list[FitRow] = []
    print(
        f"torch={torch.__version__} device={device} operator={args.operator} n={args.n} "
        f"samples={args.samples} steps={args.steps} true_alpha={args.true_alpha:g}"
    )
    for method in methods:
        params = params0.copy()
        first = np.zeros_like(params)
        second = np.zeros_like(params)
        for iteration in range(args.iterations + 1):
            nll, data_value, exact_data_value, logdet_value, gradient, seconds, exact_nll, exact_grad, rel_grad_error, rel_value_error = (
                evaluate_method(method, op, samples, params, center, args, device)
            )
            rel_param_error = float(np.linalg.norm(params - true_params) / max(np.linalg.norm(true_params), 1.0))
            rows.append(
                FitRow(
                    method,
                    iteration,
                    seconds,
                    nll,
                    exact_nll,
                    data_value,
                    exact_data_value,
                    logdet_value,
                    params[0],
                    params[1],
                    params[2],
                    math.exp(params[0]),
                    math.exp(params[1]),
                    gradient[0],
                    gradient[1],
                    gradient[2],
                    exact_grad[0],
                    exact_grad[1],
                    exact_grad[2],
                    rel_grad_error,
                    rel_value_error,
                    rel_param_error,
                    "exact logdet shared across methods; method controls data-quadratic gradient",
                )
            )
            print(
                f"{method:9s} iter={iteration:02d} nll={nll:.8e} exact_nll={exact_nll:.8e} "
                f"rel_grad={rel_grad_error:.3e} rel_data={rel_value_error:.3e} "
                f"mass={math.exp(params[0]):.3e} scale={math.exp(params[1]):.3e} alpha={params[2]:.4f} "
                f"seconds={seconds:.3f}"
            )
            if iteration < args.iterations:
                params, first, second = adam_update(
                    params,
                    gradient,
                    first,
                    second,
                    iteration + 1,
                    args.learning_rate,
                    args,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--operator", choices=["diagonal-clusters", "hadamard-clusters"], default="hadamard-clusters")
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--chebyshev-degree", type=int, default=4096)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--condition", type=float, default=1.0e10)
    parser.add_argument("--cluster-width", type=float, default=1.0e-8)
    parser.add_argument("--true-mass", type=float, default=1.0e-6)
    parser.add_argument("--true-scale", type=float, default=1.0)
    parser.add_argument("--true-alpha", type=float, default=0.5)
    parser.add_argument("--init-mass", type=float, default=5.0e-6)
    parser.add_argument("--init-scale", type=float, default=0.4)
    parser.add_argument("--init-alpha", type=float, default=0.9)
    parser.add_argument("--prior-mass", type=float, default=1.0e-5)
    parser.add_argument("--prior-scale", type=float, default=0.5)
    parser.add_argument("--prior-alpha", type=float, default=0.8)
    parser.add_argument("--prior-strength", type=float, default=1.0e-4)
    parser.add_argument("--logdet-scale", type=float, default=1.0)
    parser.add_argument("--mass-min", type=float, default=1.0e-12)
    parser.add_argument("--mass-max", type=float, default=10.0)
    parser.add_argument("--scale-min", type=float, default=1.0e-6)
    parser.add_argument("--scale-max", type=float, default=10.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--ritz-guard", choices=["none", "clip"], default="none")
    parser.add_argument("--full-reorth-passes", type=int, default=1)
    parser.add_argument("--beta-breakdown-tolerance", type=float, default=1.0e-30)
    parser.add_argument("--methods", default="exact,full,no,projected")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.operator == "hadamard-clusters" and args.n & (args.n - 1):
        raise ValueError("--n must be a power of two for hadamard-clusters")
    # Operator expects these aliases for spectral bounds.
    args.mass = args.init_mass
    args.scale = args.init_scale
    rows = run(args)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FitRow.__dataclass_fields__))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
