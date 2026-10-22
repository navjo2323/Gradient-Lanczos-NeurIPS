#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import jax
import jax.numpy as jnp
from jax import lax


def _try_add_repo_src() -> pathlib.Path | None:
    here = pathlib.Path(__file__).resolve()
    candidates: list[pathlib.Path] = []
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / "src")
        candidates.append(parent.parent / "src")
    for src in candidates:
        if (src / "matfree").exists() or (src / "matfree_extensions").exists():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return src
    return None


_REPO_SRC = _try_add_repo_src()
try:
    from matfree import decomp as repo_decomp
except Exception as exc:  # pragma: no cover
    repo_decomp = None
    _MATFREE_IMPORT_ERROR = repr(exc)
else:
    _MATFREE_IMPORT_ERROR = None

Array = jax.Array
PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class GPParams:
    log_lengthscale: Array
    log_outputscale: Array
    log_noise: Array
    mean: Array

    def tree_flatten(self):
        return (self.log_lengthscale, self.log_outputscale, self.log_noise, self.mean), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@dataclasses.dataclass
class Config:
    n: int
    d: int
    depths: list[int]
    num_probes: int
    seed: int
    jitter: float
    reference: str
    reference_depth: int
    dense_threshold: int
    outdir: str
    row_block_size: int
    timing_repeats: int
    stable_log_divdiff: bool
    clip_ritz_to_gp_floor: bool


def init_params(dtype=jnp.float64) -> GPParams:
    return GPParams(
        log_lengthscale=jnp.array(math.log(0.9), dtype=dtype),
        log_outputscale=jnp.array(math.log(1.1), dtype=dtype),
        log_noise=jnp.array(math.log(0.15), dtype=dtype),
        mean=jnp.array(0.0, dtype=dtype),
    )


def make_toy_regression(key: Array, n: int, d: int) -> tuple[Array, Array]:
    key_x, key_w, key_eps = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (n, d), dtype=jnp.float64)
    w = jax.random.normal(key_w, (d,), dtype=jnp.float64)
    y_clean = jnp.sin(x @ w) + 0.25 * jnp.cos(2.0 * x[:, 0])
    y = y_clean + 0.05 * jax.random.normal(key_eps, (n,), dtype=jnp.float64)
    return x, y


def make_rademacher_probes(key: Array, num_probes: int, n: int, dtype=jnp.float64) -> Array:
    bits = jax.random.bernoulli(key, 0.5, shape=(num_probes, n))
    return jnp.where(bits, jnp.array(1.0, dtype=dtype), jnp.array(-1.0, dtype=dtype))


@jax.jit
def kernel_matrix(params: GPParams, x: Array, jitter: float = 1e-6) -> Array:
    ell = jnp.exp(params.log_lengthscale)
    sigma_f2 = jnp.exp(2.0 * params.log_outputscale)
    sigma_n2 = jnp.exp(2.0 * params.log_noise)
    x_scaled = x / ell
    sqdist = jnp.sum((x_scaled[:, None, :] - x_scaled[None, :, :]) ** 2, axis=-1)
    K = sigma_f2 * jnp.exp(-0.5 * sqdist)
    n = x.shape[0]
    return K + (sigma_n2 + jitter) * jnp.eye(n, dtype=x.dtype)


def kernel_matmat_blocked(params: GPParams, x: Array, M: Array, *, jitter: float, row_block_size: int) -> Array:
    """Matrix-free/blockwise RBF kernel matrix-matrix product.

    Computes K(theta) @ M without materializing K. The largest dense kernel block is
    roughly (row_block_size, n), so this remains usable when n is too large for a dense K.
    """
    if M.ndim == 1:
        M = M[:, None]
        squeeze = True
    else:
        squeeze = False

    n, d = x.shape
    b = int(row_block_size)
    if b <= 0:
        raise ValueError("row_block_size must be positive")
    pad = (-n) % b

    x_pad = jnp.pad(x, ((0, pad), (0, 0))) if pad else x
    M_pad = jnp.pad(M, ((0, pad), (0, 0))) if pad else M
    n_pad = x_pad.shape[0]

    ell = jnp.exp(params.log_lengthscale)
    sigma_f2 = jnp.exp(2.0 * params.log_outputscale)
    sigma_n2 = jnp.exp(2.0 * params.log_noise)
    x_scaled = x_pad / ell

    starts = jnp.arange(0, n_pad, b)

    def one_block(start):
        xb = lax.dynamic_slice(x_scaled, (start, 0), (b, d))
        sqdist = jnp.sum((xb[:, None, :] - x_scaled[None, :, :]) ** 2, axis=-1)
        Kb = sigma_f2 * jnp.exp(-0.5 * sqdist)
        return Kb @ M_pad

    out = lax.map(one_block, starts).reshape(n_pad, M.shape[1])[:n]
    out = out + (sigma_n2 + jitter) * M
    return out[:, 0] if squeeze else out


def tree_has_nan(tree: PyTree) -> bool:
    return bool(any(bool(jnp.any(jnp.isnan(x))) for x in jax.tree_util.tree_leaves(tree)))


def tree_l2_norm(tree: PyTree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def tree_l2_error(a: PyTree, b: PyTree) -> float:
    if tree_has_nan(a) or tree_has_nan(b):
        return float("nan")
    diff = jax.tree_util.tree_map(lambda x, y: x - y, a, b)
    return tree_l2_norm(diff)


def safe_rel_grad_error(g: PyTree, g_ref: PyTree) -> float:
    return tree_l2_error(g, g_ref) / (tree_l2_norm(g_ref) + 1e-16)


def gp_to_dict(g: GPParams) -> dict[str, float]:
    return {
        "log_lengthscale": float(g.log_lengthscale),
        "log_outputscale": float(g.log_outputscale),
        "log_noise": float(g.log_noise),
        "mean": float(g.mean),
    }


def _block_until_ready_tree(obj: Any) -> None:
    """Synchronize JAX work before stopping the timer."""
    leaves = jax.tree_util.tree_leaves(obj)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def timed_call(fn: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
    """Time a callable, synchronizing JAX outputs each repeat.

    Returns the output from the last repeat and a small timing summary. Use
    --timing-repeats > 1 for more stable numbers; the mean is over repeats.
    """
    repeats = max(1, int(repeats))
    out = None
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _block_until_ready_tree(out)
        times.append(time.perf_counter() - t0)
    assert out is not None
    return out, {
        "mean_sec": float(sum(times) / len(times)),
        "min_sec": float(min(times)),
        "max_sec": float(max(times)),
        "repeats": repeats,
    }


def _normalise_decomp_result(result: Any, n: int) -> tuple[Array, Array, Any, Any]:
    """Extract (Q, T, residual, init_length_inv) from Matfree decomposition output.

    Matfree returns a NamedTuple-like object in recent versions. This helper also accepts
    a plain tuple for older/local versions. Q is returned as shape (n, m).
    """
    if hasattr(result, "Q_tall") and hasattr(result, "J_small"):
        Q = result.Q_tall
        T = result.J_small
        residual = getattr(result, "residual", None)
        init_length_inv = getattr(result, "init_length_inv", None)
    else:
        Q, T, residual, init_length_inv = result

    if isinstance(Q, tuple):
        # Bidiagonalisation returns a pair; tridiag_sym should not, but keep error clear.
        raise TypeError(f"Expected one Krylov basis from tridiag_sym, got tuple with length {len(Q)}")

    Q = jnp.asarray(Q)
    T = jnp.asarray(T)
    if Q.ndim != 2:
        raise ValueError(f"Unexpected Q shape from repo tridiag_sym: {Q.shape}")
    if Q.shape[0] == n:
        Q_nm = Q
    elif Q.shape[1] == n:
        Q_nm = Q.T
    else:
        raise ValueError(f"Cannot orient Q={Q.shape} for n={n}")
    return Q_nm, T, residual, init_length_inv


def repo_tridiag_forward(
    params: GPParams,
    x: Array,
    v: Array,
    config: Config,
    depth: int,
    *,
    reortho: str,
    custom_vjp: bool,
) -> tuple[Array, Array, Any, Any]:
    if repo_decomp is None:
        raise ImportError(f"Could not import matfree.decomp: {_MATFREE_IMPORT_ERROR}")

    decompose = repo_decomp.tridiag_sym(
        depth,
        materialize=True,
        reortho=reortho,
        custom_vjp=custom_vjp,
    )

    def matvec(vec: Array, p: GPParams) -> Array:
        return kernel_matmat_blocked(p, x, vec, jitter=config.jitter, row_block_size=config.row_block_size)

    result = decompose(matvec, v, params)
    return _normalise_decomp_result(result, n=x.shape[0])


def gp_spectral_floor(params: GPParams, config: Config) -> Array:
    """Known lower spectral bound for K = kernel + (noise^2+jitter) I."""
    return jnp.exp(2.0 * params.log_noise) + jnp.asarray(config.jitter, dtype=params.log_noise.dtype)


def _maybe_clip_ritz(evals: Array, floor: Array | None) -> Array:
    tiny = jnp.finfo(evals.dtype).tiny
    if floor is None:
        return jnp.maximum(evals, tiny)
    return jnp.maximum(evals, jnp.maximum(tiny, floor.astype(evals.dtype)))


def quad_from_T(T: Array, probe_norm_sq: Array, *, ritz_floor: Array | None = None) -> Array:
    evals, evecs = jnp.linalg.eigh(T)
    # Protect log from tiny/negative Ritz values due to roundoff or no-reorth ghosts.
    evals = _maybe_clip_ritz(evals, ritz_floor)
    c = evecs[0, :]
    return probe_norm_sq * jnp.sum((c ** 2) * jnp.log(evals))


def _stable_log_divided_difference(evals: Array) -> Array:
    """Stable symmetric matrix F_ij=(log a-log b)/(a-b), F_ii=1/a.

    Uses d=(a-b)/(a+b) and
        (log a-log b)/(a-b) = (2/(a+b)) * atanh(d)/d.
    For small d, atanh(d)/d is evaluated by its series.
    This avoids cancellation when Ritz values are close and is exactly symmetric.
    """
    a = evals[:, None]
    b = evals[None, :]
    denom = a + b
    d = (a - b) / denom
    d2 = d * d
    ratio_series = 1.0 + d2 / 3.0 + (d2 * d2) / 5.0
    ratio_exact = jnp.arctanh(d) / d
    ratio = jnp.where(jnp.abs(d) < 1e-4, ratio_series, ratio_exact)
    return (2.0 / denom) * ratio


def divided_difference_log_matrix(
    T: Array,
    probe_norm_sq: Array,
    *,
    stable: bool = True,
    ritz_floor: Array | None = None,
) -> Array:
    evals, evecs = jnp.linalg.eigh(T)
    evals = _maybe_clip_ritz(evals, ritz_floor)
    if stable:
        F = _stable_log_divided_difference(evals)
    else:
        logevals = jnp.log(evals)
        li = logevals[:, None]
        lj = logevals[None, :]
        ei = evals[:, None]
        ej = evals[None, :]
        diff_e = ei - ej
        diff_l = li - lj
        # Relative tolerance instead of absolute tolerance.
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(ei), jnp.abs(ej)))
        F = jnp.where(jnp.abs(diff_e) > 1e-12 * scale, diff_l / diff_e, 1.0 / ei)
        F = 0.5 * (F + F.T)
    c = evecs[0, :]
    mid = (c[:, None] * c[None, :]) * F
    G = probe_norm_sq * (evecs @ mid @ evecs.T)
    return 0.5 * (G + G.T)


def repo_forward_value_and_frozen_VW(
    params: GPParams,
    x: Array,
    probes: Array,
    config: Config,
    depth: int,
    *,
    reortho: str,
) -> tuple[float, Array, Array]:
    values = []
    Vs = []
    Ws = []
    ritz_floor = gp_spectral_floor(params, config) if config.clip_ritz_to_gp_floor else None
    for v in probes:
        Q, T, _residual, _cinv = repo_tridiag_forward(params, x, v, config, depth, reortho=reortho, custom_vjp=False)
        probe_norm_sq = jnp.sum(v ** 2)
        values.append(quad_from_T(T, probe_norm_sq, ritz_floor=ritz_floor))
        G = divided_difference_log_matrix(
            T,
            probe_norm_sq,
            stable=config.stable_log_divdiff,
            ritz_floor=ritz_floor,
        )
        V = lax.stop_gradient(Q)
        W = lax.stop_gradient(V @ G)
        Vs.append(V)
        Ws.append(W)
    value = jnp.mean(jnp.stack(values))
    return float(value), jnp.stack(Vs), jnp.stack(Ws)


def ours_grad_from_frozen_VW(params: GPParams, x: Array, Vs: Array, Ws: Array, config: Config) -> GPParams:
    p, n, m = Vs.shape

    def total_obj(par: GPParams) -> Array:
        V_block = jnp.transpose(Vs, (1, 0, 2)).reshape(n, p * m)
        KV_block = kernel_matmat_blocked(par, x, V_block, jitter=config.jitter, row_block_size=config.row_block_size)
        KVs = jnp.transpose(KV_block.reshape(n, p, m), (1, 0, 2))
        return jnp.mean(jnp.sum(Ws * KVs, axis=(1, 2)))

    g = jax.grad(total_obj)(params)
    return GPParams(g.log_lengthscale, g.log_outputscale, g.log_noise, jnp.array(0.0, dtype=x.dtype))


def repo_value_for_adjoint(params: GPParams, x: Array, probes: Array, config: Config, depth: int, *, reortho: str) -> Array:
    values = []
    for v in probes:
        _Q, T, _residual, _cinv = repo_tridiag_forward(params, x, v, config, depth, reortho=reortho, custom_vjp=True)
        values.append(quad_from_T(T, jnp.sum(v ** 2)))
    return jnp.mean(jnp.stack(values))


def repo_adjoint_value_and_grad(
    params: GPParams,
    x: Array,
    probes: Array,
    config: Config,
    depth: int,
    *,
    reortho: str,
) -> tuple[float, GPParams]:
    value, grad = jax.value_and_grad(lambda p: repo_value_for_adjoint(p, x, probes, config, depth, reortho=reortho))(params)
    grad = GPParams(grad.log_lengthscale, grad.log_outputscale, grad.log_noise, jnp.array(0.0, dtype=x.dtype))
    return float(value), grad


def dense_value_and_grad(params: GPParams, x: Array, probes: Array, config: Config) -> tuple[float, GPParams]:
    """Dense exact fixed-probe reference without differentiating through eigh."""
    K = np.asarray(kernel_matrix(params, x, jitter=config.jitter))
    evals, U = np.linalg.eigh(K)
    evals = np.maximum(evals, np.finfo(evals.dtype).tiny)

    probes_np = np.asarray(probes)
    B = (probes_np.T @ probes_np) / probes_np.shape[0]
    C = U.T @ B @ U
    logevals = np.log(evals)
    diff_e = evals[:, None] - evals[None, :]
    diff_l = logevals[:, None] - logevals[None, :]
    F = np.empty_like(diff_e)
    mask = np.abs(diff_e) > 1e-12
    F[mask] = diff_l[mask] / diff_e[mask]
    diag_or_close = np.broadcast_to(1.0 / evals[:, None], diff_e.shape)
    F[~mask] = diag_or_close[~mask]
    S = U @ (F * C) @ U.T
    S = 0.5 * (S + S.T)

    values = np.sum((probes_np @ U) ** 2 * logevals[None, :], axis=1)
    value = float(np.mean(values))
    S_jax = jnp.asarray(S, dtype=x.dtype)

    def linearized_dense_objective(p: GPParams) -> Array:
        Kp = kernel_matrix(p, x, jitter=config.jitter)
        return jnp.sum(S_jax * Kp)

    grad = jax.grad(linearized_dense_objective)(params)
    grad = GPParams(grad.log_lengthscale, grad.log_outputscale, grad.log_noise, jnp.array(0.0, dtype=x.dtype))
    return value, grad


def make_reference(params: GPParams, x: Array, probes: Array, config: Config) -> tuple[str, float, GPParams]:
    if config.reference == "dense" or (config.reference == "auto" and config.n <= config.dense_threshold):
        value, grad = dense_value_and_grad(params, x, probes, config)
        return "dense eigendecomposition", value, grad

    if config.reference in {"auto", "full_high_m"}:
        value, Vs, Ws = repo_forward_value_and_frozen_VW(
            params, x, probes, config, config.reference_depth, reortho="full"
        )
        grad = ours_grad_from_frozen_VW(params, x, Vs, Ws, config)
        return f"ours from repo full reorth forward at m_ref={config.reference_depth}", value, grad

    if config.reference == "none_high_m":
        value, Vs, Ws = repo_forward_value_and_frozen_VW(
            params, x, probes, config, config.reference_depth, reortho="none"
        )
        grad = ours_grad_from_frozen_VW(params, x, Vs, Ws, config)
        return f"ours from repo no reorth forward at m_ref={config.reference_depth}", value, grad

    raise ValueError(f"Unknown reference: {config.reference}")


def rel_scalar_error(value: float, ref: float) -> float:
    if math.isnan(value) or math.isnan(ref):
        return float("nan")
    return abs(value - ref) / (abs(ref) + 1e-16)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Compare our frozen-forward gradient vs repo Lanczos adjoint; no GPyTorch.")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--depths", type=str, default="10,20,30,40,50,60,80,100")
    p.add_argument("--num-probes", type=int, default=1, help="1 gives a single w^T log(K) w; >1 gives fixed-probe average.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=1e-6)
    p.add_argument("--reference", choices=["auto", "dense", "full_high_m", "none_high_m"], default="auto")
    p.add_argument("--reference-depth", type=int, default=300)
    p.add_argument("--dense-threshold", type=int, default=2500)
    p.add_argument("--outdir", type=str, default="outputs/matrix_log")
    p.add_argument("--row-block-size", type=int, default=512, help="Rows of K materialized per matvec block.")
    p.add_argument("--timing-repeats", type=int, default=1, help="Repeat each timed call and store mean/min/max seconds.")
    p.add_argument("--unstable-log-divdiff", action="store_true", help="Use the original naive log divided-difference formula instead of the stable atanh formula.")
    p.add_argument("--clip-ritz-to-gp-floor", action="store_true", help="Clip small tridiagonal Ritz values to the known GP spectral floor noise^2+jitter before log and G construction. This stabilizes no-reorth ghosts but changes the small projected objective.")
    a = p.parse_args()
    depths = [int(s.strip()) for s in a.depths.split(",") if s.strip()]
    return Config(
        n=a.n,
        d=a.d,
        depths=depths,
        num_probes=a.num_probes,
        seed=a.seed,
        jitter=a.jitter,
        reference=a.reference,
        reference_depth=a.reference_depth,
        dense_threshold=a.dense_threshold,
        outdir=a.outdir,
        row_block_size=a.row_block_size,
        timing_repeats=a.timing_repeats,
        stable_log_divdiff=not a.unstable_log_divdiff,
        clip_ritz_to_gp_floor=a.clip_ritz_to_gp_floor,
    )


def plot_results(rows: list[dict[str, Any]], out_png: pathlib.Path, title_suffix: str) -> None:
    ms = np.array([r["m"] for r in rows], dtype=float)

    fwd_full = np.array([r["forward"]["full"]["rel_error"] for r in rows], dtype=float)
    fwd_none = np.array([r["forward"]["none"]["rel_error"] for r in rows], dtype=float)

    grad_ours_full = np.array([r["gradients"]["ours_full"]["rel_error_l2"] for r in rows], dtype=float)
    grad_adj_full = np.array([r["gradients"]["adjoint_full"]["rel_error_l2"] for r in rows], dtype=float)
    grad_ours_none = np.array([r["gradients"]["ours_none"]["rel_error_l2"] for r in rows], dtype=float)
    grad_adj_none = np.array([r["gradients"]["adjoint_none"]["rel_error_l2"] for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 3.9), sharex=True)

    ax0.semilogy(ms, fwd_full, marker="o", label="Forward: full reorth.")
    ax0.semilogy(ms, fwd_none, marker="v", label="Forward: no reorth.")
    ax0.set_title(r"Forward error: fixed-probe $v^T\log(K)v$")
    ax0.set_xlabel("Lanczos iterations m")
    ax0.set_ylabel("Relative forward error")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.semilogy(ms, grad_ours_full, marker="o", linestyle="-", label="Ours, full forward")
    ax1.semilogy(ms, grad_adj_full, marker="o", linestyle="--", label="Adjoint, full forward")
    ax1.semilogy(ms, grad_ours_none, marker="v", linestyle="-", label="Ours, no-reorth forward")
    ax1.semilogy(ms, grad_adj_none, marker="v", linestyle="--", label="Adjoint, no-reorth forward")
    ax1.set_title(r"Gradient error: $\nabla_\theta v^T\log(K)v$")
    ax1.set_xlabel("Lanczos iterations m")
    ax1.set_ylabel("Relative gradient error")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_timing_results(rows: list[dict[str, Any]], out_png: pathlib.Path, title_suffix: str) -> None:
    ms = np.array([r["m"] for r in rows], dtype=float)
    x = np.arange(len(rows))

    fwd_full = np.array([r["timings"]["full"]["forward"]["mean_sec"] for r in rows], dtype=float)
    fwd_none = np.array([r["timings"]["none"]["forward"]["mean_sec"] for r in rows], dtype=float)

    ours_full_total = np.array([r["timings"]["full"]["ours_total"]["mean_sec"] for r in rows], dtype=float)
    adj_full_total = np.array([r["timings"]["full"]["adjoint_value_and_grad"]["mean_sec"] for r in rows], dtype=float)
    ours_none_total = np.array([r["timings"]["none"]["ours_total"]["mean_sec"] for r in rows], dtype=float)
    adj_none_total = np.array([r["timings"]["none"]["adjoint_value_and_grad"]["mean_sec"] for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)

    width0 = 0.34
    ax0.bar(x - width0 / 2, fwd_full, width0, label="Forward full reorth.")
    ax0.bar(x + width0 / 2, fwd_none, width0, label="Forward no reorth.")
    ax0.set_title("Forward runtime")
    ax0.set_xlabel("Lanczos iterations m")
    ax0.set_ylabel("Time (s)")
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(int(m)) for m in ms])
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(fontsize=8)

    width1 = 0.20
    ax1.bar(x - 1.5 * width1, ours_full_total, width1, label="Ours total, full")
    ax1.bar(x - 0.5 * width1, adj_full_total, width1, label="Adjoint total, full")
    ax1.bar(x + 0.5 * width1, ours_none_total, width1, label="Ours total, none")
    ax1.bar(x + 1.5 * width1, adj_none_total, width1, label="Adjoint total, none")
    ax1.set_title("Value + gradient runtime")
    ax1.set_xlabel("Lanczos iterations m")
    ax1.set_ylabel("Time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(m)) for m in ms])
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _json_finite(value):
    """Represent a failed/nonfinite measurement as null in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    return value


def main() -> None:
    config = parse_args()
    jax.config.update("jax_enable_x64", True)

    if repo_decomp is None:
        raise ImportError(f"Could not import matfree.decomp. repo_src={_REPO_SRC}, error={_MATFREE_IMPORT_ERROR}")

    key = jax.random.PRNGKey(config.seed)
    key_data, key_probe = jax.random.split(key)
    x, _y = make_toy_regression(key_data, config.n, config.d)
    params = init_params(dtype=x.dtype)
    probes = make_rademacher_probes(key_probe, config.num_probes, config.n, dtype=x.dtype)

    ref_name, ref_value, ref_grad = make_reference(params, x, probes, config)
    print(f"Reference: {ref_name}, value={ref_value:.12e}, ||grad||={tree_l2_norm(ref_grad):.6e}", flush=True)

    rows: list[dict[str, Any]] = []
    for depth in config.depths:
        print(f"[m={depth}]", flush=True)
        row: dict[str, Any] = {"m": depth, "forward": {}, "gradients": {}, "timings": {}}

        for reortho in ["full", "none"]:
            try:
                (value, Vs, Ws), forward_timing = timed_call(
                    lambda: repo_forward_value_and_frozen_VW(params, x, probes, config, depth, reortho=reortho),
                    config.timing_repeats,
                )
                ours_grad, ours_grad_timing = timed_call(
                    lambda: ours_grad_from_frozen_VW(params, x, Vs, Ws, config),
                    config.timing_repeats,
                )
                (adj_value, adj_grad), adjoint_timing = timed_call(
                    lambda: repo_adjoint_value_and_grad(params, x, probes, config, depth, reortho=reortho),
                    config.timing_repeats,
                )
                fwd_rel = rel_scalar_error(value, ref_value)
                ours_rel = safe_rel_grad_error(ours_grad, ref_grad)
                adj_rel = safe_rel_grad_error(adj_grad, ref_grad)
                adj_fwd_rel = rel_scalar_error(adj_value, value)
                row["forward"][reortho] = {
                    "value": value,
                    "rel_error": fwd_rel,
                    "adjoint_forward_value": adj_value,
                    "adjoint_forward_rel_diff_vs_frozen_forward": adj_fwd_rel,
                }
                row["gradients"][f"ours_{reortho}"] = {
                    "grad": gp_to_dict(ours_grad),
                    "rel_error_l2": ours_rel,
                    "error": None,
                }
                row["gradients"][f"adjoint_{reortho}"] = {
                    "grad": gp_to_dict(adj_grad),
                    "rel_error_l2": adj_rel,
                    "error": None,
                }
                ours_total_timing = {
                    "mean_sec": float(forward_timing["mean_sec"] + ours_grad_timing["mean_sec"]),
                    "min_sec": float(forward_timing["min_sec"] + ours_grad_timing["min_sec"]),
                    "max_sec": float(forward_timing["max_sec"] + ours_grad_timing["max_sec"]),
                    "repeats": config.timing_repeats,
                    "definition": "forward custom_vjp=False plus frozen-VW gradient",
                }
                row["timings"][reortho] = {
                    "forward": forward_timing,
                    "ours_grad_from_frozen_VW": ours_grad_timing,
                    "ours_total": ours_total_timing,
                    "adjoint_value_and_grad": adjoint_timing,
                }
                print(
                    f"  {reortho:4s} forward_rel={fwd_rel:.3e} "
                    f"ours_grad_rel={ours_rel:.3e} adjoint_grad_rel={adj_rel:.3e} "
                    f"adjoint_forward_diff={adj_fwd_rel:.3e} "
                    f"t_forward={forward_timing['mean_sec']:.3e}s "
                    f"t_ours_total={ours_total_timing['mean_sec']:.3e}s "
                    f"t_adjoint={adjoint_timing['mean_sec']:.3e}s",
                    flush=True,
                )
            except Exception as exc:
                nan_grad = GPParams(*(jnp.array(float("nan"), dtype=x.dtype) for _ in range(3)), jnp.array(0.0, dtype=x.dtype))
                row["forward"][reortho] = {"value": float("nan"), "rel_error": float("nan"), "error": repr(exc)}
                row["gradients"][f"ours_{reortho}"] = {"grad": gp_to_dict(nan_grad), "rel_error_l2": float("nan"), "error": repr(exc)}
                row["gradients"][f"adjoint_{reortho}"] = {"grad": gp_to_dict(nan_grad), "rel_error_l2": float("nan"), "error": repr(exc)}
                nan_timing = {"mean_sec": float("nan"), "min_sec": float("nan"), "max_sec": float("nan"), "repeats": config.timing_repeats, "error": repr(exc)}
                row["timings"][reortho] = {"forward": nan_timing, "ours_grad_from_frozen_VW": nan_timing, "ours_total": nan_timing, "adjoint_value_and_grad": nan_timing}
                print(f"  {reortho:4s} FAILED: {repr(exc)}", flush=True)

        rows.append(row)

    outdir = pathlib.Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"n{config.n}_d{config.d}_p{config.num_probes}_seed{config.seed}_ref{config.reference}"
    out_json = outdir / f"repo_forward_ours_vs_adjoint_{stem}.json"
    out_png = outdir / f"repo_forward_ours_vs_adjoint_{stem}.png"
    out_timing_png = outdir / f"repo_forward_ours_vs_adjoint_timings_{stem}.png"
    payload = {
        "config": dataclasses.asdict(config),
        "reference": {
            "name": ref_name,
            "value": ref_value,
            "grad": gp_to_dict(ref_grad),
            "grad_l2_norm": tree_l2_norm(ref_grad),
        },
        "rows": rows,
        "notes": [
            "No GPyTorch baseline is used.",
            "For each reorthogonalization mode, the forward value and our gradient use Matfree/repo tridiag_sym with custom_vjp=False and frozen V,W.",
            "The adjoint gradient uses the same Matfree/repo tridiag_sym forward with custom_vjp=True, so only the backward rule differs.",
            "The objective is mean_l v_l^T log(K(theta)) v_l; with num_probes=1 this is a single-vector objective.",
            "Timing fields are mean/min/max seconds over --timing-repeats, synchronized with block_until_ready.",
            "By default, G for log uses a stable symmetric atanh divided-difference formula; --unstable-log-divdiff restores the naive formula.",
            "If --clip-ritz-to-gp-floor is set, small/ghost Ritz values are clipped to noise^2+jitter before log/G construction; this stabilizes no-reorth but changes the projected small objective.",
        ],
    }
    out_json.write_text(json.dumps(_json_finite(payload), indent=2, allow_nan=False))
    title = f"n={config.n}, probes={config.num_probes}, reference={ref_name}"
    plot_results(rows, out_png, title)
    plot_timing_results(rows, out_timing_png, title)

    print(json.dumps({"saved_json": str(out_json), "saved_plot": str(out_png), "saved_timing_plot": str(out_timing_png)}, indent=2))


if __name__ == "__main__":
    main()
