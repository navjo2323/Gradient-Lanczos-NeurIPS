from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from benchmark_pkg_directions import Direction


@dataclass
class LanczosResult:
    x: list[np.ndarray]
    a: np.ndarray
    b: np.ndarray
    K_eff: int
    norm_v: float


@dataclass
class BatchedLanczosResult:
    x: list[np.ndarray]
    a: np.ndarray
    b: np.ndarray
    K_eff: list[int]
    norm_v: np.ndarray


@dataclass
class EvalResult:
    value: float
    grad_exact: float
    grad_approx: float
    runtime_total_sec: float
    runtime_forward_sec: float
    runtime_approx_grad_sec: float
    runtime_exact_grad_sec: float
    k_eff: int


def lanczos_forward(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray, K: int, tol: float = 1e-14) -> LanczosResult:
    norm_v = la.norm(v)
    if norm_v == 0.0:
        raise ValueError("Starting vector must be nonzero.")

    x_prev = np.zeros_like(v)
    x_curr = v / norm_v
    beta_prev = 1.0

    x_list = [x_curr.copy()]
    a = []
    b = []

    for _ in range(K):
        w = matvec(x_curr) - beta_prev * x_prev
        alpha = float(np.dot(x_curr, w))
        w = w - alpha * x_curr
        beta = float(la.norm(w))
        a.append(alpha)
        b.append(beta)
        if beta < tol:
            x_list.append(np.zeros_like(v))
            return LanczosResult(x=x_list, a=np.array(a), b=np.array(b), K_eff=len(a), norm_v=norm_v)
        x_next = w / beta
        x_list.append(x_next.copy())
        x_prev = x_curr
        x_curr = x_next
        beta_prev = beta

    return LanczosResult(x=x_list, a=np.array(a), b=np.array(b), K_eff=K, norm_v=norm_v)


def lanczos_forward_batched(
    matmat: Callable[[np.ndarray], np.ndarray],
    V: np.ndarray,
    K: int,
    tol: float = 1e-14,
) -> BatchedLanczosResult:
    norm_v = la.norm(V, axis=0)
    if np.any(norm_v == 0.0):
        raise ValueError("Starting vectors must be nonzero.")

    n, batch = V.shape
    x_prev = np.zeros((n, batch), dtype=V.dtype)
    x_curr = V / norm_v[None, :]
    beta_prev = np.ones(batch, dtype=V.dtype)

    x_list = [x_curr.copy()]
    a = np.zeros((batch, K), dtype=float)
    b = np.zeros((batch, K), dtype=float)
    K_eff = [K] * batch
    active = np.ones(batch, dtype=bool)

    for k in range(K):
        w = matmat(x_curr) - x_prev * beta_prev[None, :]
        alpha = np.sum(x_curr * w, axis=0)
        w = w - x_curr * alpha[None, :]
        beta = la.norm(w, axis=0)
        a[:, k] = alpha
        b[:, k] = beta

        newly_done = active & (beta < tol)
        for col in np.flatnonzero(newly_done):
            K_eff[col] = k + 1
        active &= ~newly_done

        beta_safe = np.where(beta < tol, 1.0, beta)
        x_next = w / beta_safe[None, :]
        x_next[:, ~active] = 0.0
        x_list.append(x_next.copy())
        x_prev = x_curr
        x_curr = x_next
        beta_prev = beta

        if not np.any(active):
            break

    return BatchedLanczosResult(x=x_list, a=a, b=b, K_eff=K_eff, norm_v=norm_v)


def build_tridiag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    K = len(a)
    T = np.diag(a)
    if K > 1:
        off = b[: K - 1]
        T += np.diag(off, 1) + np.diag(off, -1)
    return T


def divided_difference_exp(evals: np.ndarray) -> np.ndarray:
    g = np.exp(evals)
    diff = evals[:, None] - evals[None, :]
    same = np.isclose(diff, 0.0)
    safe = np.where(same, 1.0, diff)
    F = (g[:, None] - g[None, :]) / safe
    np.fill_diagonal(F, g)
    return F


def small_matrix_gradient_exp(T: np.ndarray, norm_v_sq: float) -> np.ndarray:
    evals, Q = la.eigh(T)
    e1 = np.zeros(T.shape[0])
    e1[0] = 1.0
    c = Q.T @ e1
    F = divided_difference_exp(evals)
    H = np.outer(c, c) * F
    G = norm_v_sq * (Q @ H @ Q.T)
    return 0.5 * (G + G.T)


def evaluate_quadratic_with_both_gradients(
    A0: sp.csr_matrix,
    direction: Direction,
    theta: float,
    u: np.ndarray,
    K: int,
    time_scale: float = 1.0,
) -> EvalResult:
    t_total0 = perf_counter()

    E_mat = direction.maybe_matrix()
    if E_mat is None:
        A_theta = A0.copy()
    else:
        A_theta = (A0 + theta * E_mat).tocsr()
    At = (time_scale * A_theta).tocsr()

    def matvec(x: np.ndarray) -> np.ndarray:
        return At @ x

    t0 = perf_counter()
    out = lanczos_forward(matvec, u, K)
    runtime_forward_sec = perf_counter() - t0

    K_eff = out.K_eff
    T = build_tridiag(out.a[:K_eff], out.b[:K_eff])

    e1 = np.zeros(K_eff)
    e1[0] = 1.0
    value = (out.norm_v ** 2) * float(e1 @ la.expm(T) @ e1)
    G = small_matrix_gradient_exp(T, out.norm_v ** 2)

    # Cheap projected gradient path: <V G V^T, E> = <G, V^T E V>
    # Do not materialize the dense n x k matrix V.
    t0 = perf_counter()
    small_E = direction.projected_from_basis(out.x, K_eff)
    grad_approx = float(np.sum(G * small_E)) * time_scale
    runtime_approx_grad_sec = perf_counter() - t0

    # Exact backward Lanczos adjoint
    t0 = perf_counter()
    grad_a = np.diag(G).copy()
    grad_b = np.zeros(K_eff)
    if K_eff > 1:
        grad_b[: K_eff - 1] = 2.0 * np.diag(G, 1)
    grad_b[K_eff - 1] = 0.0

    x = out.x
    lambdas = [np.zeros_like(u) for _ in range(K_eff + 2)]
    zetas = [np.zeros_like(u) for _ in range(K_eff + 2)]

    for k in range(K_eff, 0, -1):
        idx = k - 1
        bk = out.b[idx]
        ak = out.a[idx]
        xk = x[idx]
        xk1 = x[idx + 1]

        xi = np.zeros_like(u) if bk == 0.0 else zetas[k + 1] / bk
        mu_tilde = grad_b[idx] - float(np.dot(lambdas[k + 1], xk)) + float(np.dot(xk1, xi))
        nu_tilde = grad_a[idx] + float(np.dot(xk, xi))

        lambdas[k] = -xi + mu_tilde * xk1 + nu_tilde * xk
        zetas[k] = -(At @ lambdas[k]) + ak * lambdas[k] + bk * lambdas[k + 1] - bk * nu_tilde * xk1

    grad_exact = 0.0
    for k in range(1, K_eff + 1):
        grad_exact += direction.contraction(x[k - 1], lambdas[k])
    grad_exact *= time_scale
    runtime_exact_grad_sec = perf_counter() - t0

    runtime_total_sec = perf_counter() - t_total0
    return EvalResult(
        value=value,
        grad_exact=float(grad_exact),
        grad_approx=float(grad_approx),
        runtime_total_sec=runtime_total_sec,
        runtime_forward_sec=runtime_forward_sec,
        runtime_approx_grad_sec=runtime_approx_grad_sec,
        runtime_exact_grad_sec=runtime_exact_grad_sec,
        k_eff=K_eff,
    )


def _quadratic_from_existing_lanczos(
    At: sp.csr_matrix,
    direction: Direction,
    u: np.ndarray,
    K_eff: int,
    norm_v: float,
    x: list[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    time_scale: float,
) -> tuple[float, float, float, float, float]:
    T = build_tridiag(a[:K_eff], b[:K_eff])

    e1 = np.zeros(K_eff)
    e1[0] = 1.0
    value = (norm_v ** 2) * float(e1 @ la.expm(T) @ e1)
    G = small_matrix_gradient_exp(T, norm_v ** 2)

    t0 = perf_counter()
    small_E = direction.projected_from_basis(x, K_eff)
    grad_approx = float(np.sum(G * small_E)) * time_scale
    runtime_approx_grad_sec = perf_counter() - t0

    t0 = perf_counter()
    grad_a = np.diag(G).copy()
    grad_b = np.zeros(K_eff)
    if K_eff > 1:
        grad_b[: K_eff - 1] = 2.0 * np.diag(G, 1)
    grad_b[K_eff - 1] = 0.0

    lambdas = [np.zeros_like(u) for _ in range(K_eff + 2)]
    zetas = [np.zeros_like(u) for _ in range(K_eff + 2)]

    for k in range(K_eff, 0, -1):
        idx = k - 1
        bk = b[idx]
        ak = a[idx]
        xk = x[idx]
        xk1 = x[idx + 1]

        xi = np.zeros_like(u) if bk == 0.0 else zetas[k + 1] / bk
        mu_tilde = grad_b[idx] - float(np.dot(lambdas[k + 1], xk)) + float(np.dot(xk1, xi))
        nu_tilde = grad_a[idx] + float(np.dot(xk, xi))

        lambdas[k] = -xi + mu_tilde * xk1 + nu_tilde * xk
        zetas[k] = -(At @ lambdas[k]) + ak * lambdas[k] + bk * lambdas[k + 1] - bk * nu_tilde * xk1

    grad_exact = 0.0
    for k in range(1, K_eff + 1):
        grad_exact += direction.contraction(x[k - 1], lambdas[k])
    grad_exact *= time_scale
    runtime_exact_grad_sec = perf_counter() - t0

    return (
        float(value),
        float(grad_exact),
        float(grad_approx),
        runtime_approx_grad_sec,
        runtime_exact_grad_sec,
    )


def evaluate_bilinear_with_both_gradients(
    A0: sp.csr_matrix,
    direction: Direction,
    theta: float,
    u: np.ndarray,
    v: np.ndarray,
    K: int,
    time_scale: float = 1.0,
) -> EvalResult:
    t_total0 = perf_counter()

    E_mat = direction.maybe_matrix()
    if E_mat is None:
        A_theta = A0.copy()
    else:
        A_theta = (A0 + theta * E_mat).tocsr()
    At = (time_scale * A_theta).tocsr()

    starts = np.column_stack([u + v, u - v])

    t0 = perf_counter()
    out = lanczos_forward_batched(lambda X: At @ X, starts, K)
    runtime_forward_sec = perf_counter() - t0

    values = []
    grad_exact = []
    grad_approx = []
    runtime_approx = 0.0
    runtime_exact = 0.0
    for col in range(starts.shape[1]):
        K_eff_col = out.K_eff[col]
        basis = [basis_block[:, col] for basis_block in out.x]
        value_col, grad_exact_col, grad_approx_col, approx_sec, exact_sec = _quadratic_from_existing_lanczos(
            At,
            direction,
            starts[:, col],
            K_eff_col,
            float(out.norm_v[col]),
            basis,
            out.a[col, :],
            out.b[col, :],
            time_scale,
        )
        values.append(value_col)
        grad_exact.append(grad_exact_col)
        grad_approx.append(grad_approx_col)
        runtime_approx += approx_sec
        runtime_exact += exact_sec

    return EvalResult(
        value=0.25 * (values[0] - values[1]),
        grad_exact=0.25 * (grad_exact[0] - grad_exact[1]),
        grad_approx=0.25 * (grad_approx[0] - grad_approx[1]),
        runtime_total_sec=perf_counter() - t_total0,
        runtime_forward_sec=runtime_forward_sec,
        runtime_approx_grad_sec=runtime_approx,
        runtime_exact_grad_sec=runtime_exact,
        k_eff=max(out.K_eff),
    )


def evaluate_trace_with_both_gradients(
    A0: sp.csr_matrix,
    direction: Direction,
    theta: float,
    K: int,
    time_scale: float,
    nprobe: int,
    seed: int,
) -> EvalResult:
    rng = np.random.default_rng(seed)
    vals = []
    ge = []
    ga = []
    rt_total = 0.0
    rt_forward = 0.0
    rt_approx = 0.0
    rt_exact = 0.0
    k_eff = 0

    for _ in range(nprobe):
        v = rng.choice([-1.0, 1.0], size=A0.shape[0])
        r = evaluate_quadratic_with_both_gradients(A0, direction, theta, v, K, time_scale)
        vals.append(r.value)
        ge.append(r.grad_exact)
        ga.append(r.grad_approx)
        rt_total += r.runtime_total_sec
        rt_forward += r.runtime_forward_sec
        rt_approx += r.runtime_approx_grad_sec
        rt_exact += r.runtime_exact_grad_sec
        k_eff = max(k_eff, r.k_eff)

    return EvalResult(
        value=float(np.mean(vals)),
        grad_exact=float(np.mean(ge)),
        grad_approx=float(np.mean(ga)),
        runtime_total_sec=rt_total,
        runtime_forward_sec=rt_forward,
        runtime_approx_grad_sec=rt_approx,
        runtime_exact_grad_sec=rt_exact,
        k_eff=k_eff,
    )
