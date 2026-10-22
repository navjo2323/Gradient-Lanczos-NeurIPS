"""Real symmetric, matrix-free projected gradients.

This small reference API implements the paper's algorithm. The experiment folders
preserve the implementations used for the reported results. The returned gradient
approximates the matrix-function derivative; it does not differentiate the finite
Lanczos recurrence. All arrays and callbacks in this API are real float64.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class LanczosResult:
    alpha: Array
    beta: Array
    norm_squared: float
    start: Array
    basis: Array | None

    @property
    def tridiagonal(self) -> Array:
        return np.diag(self.alpha) + np.diag(self.beta, 1) + np.diag(self.beta, -1)


def _real_vector(x: Array, name: str) -> Array:
    if np.iscomplexobj(x):
        raise ValueError(f"{name} must be real; use the experiment's polarization for complex objectives")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return x


def lanczos(matvec: Callable[[Array], Array], u: Array, depth: int, *,
            store_basis: bool = True, reorthogonalize: bool = False,
            breakdown_tolerance: float = 1e-14) -> LanczosResult:
    """Run a symmetric three-term recurrence, stopping at numerical breakdown.

    ``reorthogonalize=True`` applies two full orthogonalization passes and needs
    the stored basis. Replay uses the unreorthogonalized recurrence only.
    ``beta`` contains only the internal off-diagonal entries of T.
    """
    u = _real_vector(u, "u")
    if not isinstance(depth, (int, np.integer)) or not 1 <= depth <= u.size:
        raise ValueError("depth must be an integer between 1 and len(u)")
    if reorthogonalize and not store_basis:
        raise ValueError("reorthogonalization requires store_basis=True")
    if breakdown_tolerance < 0 or not np.isfinite(breakdown_tolerance):
        raise ValueError("breakdown_tolerance must be finite and nonnegative")
    norm = float(np.linalg.norm(u))
    if norm == 0:
        raise ValueError("u must be nonzero")
    v = u / norm
    start = v.copy()
    previous = np.zeros_like(v)
    alpha, beta, vectors = [], [], []
    for j in range(depth):
        if store_basis:
            vectors.append(v.copy())
        av = _real_vector(matvec(v), "matvec(v)")
        if av.shape != v.shape:
            raise ValueError("matvec(v) has the wrong shape")
        w = av - (beta[-1] * previous if j else 0.0)
        a = float(v @ w)
        w = w - a * v
        if reorthogonalize:
            for _ in range(2):
                for q in vectors:
                    w -= (q @ w) * q
        alpha.append(a)
        b = float(np.linalg.norm(w))
        if j == depth - 1 or b <= breakdown_tolerance * max(1.0, np.linalg.norm(av)):
            break
        beta.append(b)
        previous, v = v, w / b
    return LanczosResult(np.asarray(alpha), np.asarray(beta), norm * norm,
                         start, np.column_stack(vectors) if store_basis else None)


def projected_sensitivity(result: LanczosResult, f: Callable[[Array], Array],
                          df: Callable[[Array], Array]) -> tuple[float, Array]:
    """Return rho*e1' f(T) e1 and G=rho*Lf(T,e1 e1').

    Near equal eigenvalues use the derivative at their midpoint. This stabilizes
    the reference API and is distinct from experiment-specific historical rules.
    The function and derivative must be finite on the Ritz spectrum.
    """
    lam, q = np.linalg.eigh(result.tridiagonal)
    values = np.asarray(f(lam), dtype=np.float64)
    if values.shape != lam.shape or not np.all(np.isfinite(values)):
        raise ValueError("f must return finite values on the Ritz spectrum")
    delta = lam[:, None] - lam[None, :]
    scale = np.maximum(1.0, np.maximum(np.abs(lam[:, None]), np.abs(lam[None, :])))
    near = np.abs(delta) <= 8 * np.finfo(float).eps * scale
    divided = np.zeros_like(delta)
    np.divide(values[:, None] - values[None, :], delta, out=divided, where=~near)
    midpoint = 0.5 * (lam[:, None] + lam[None, :])
    divided[near] = np.asarray(df(midpoint[near]), dtype=np.float64)
    if not np.all(np.isfinite(divided)):
        raise ValueError("df must return finite values on the Ritz spectrum")
    c = q[0, :]
    g = result.norm_squared * (q @ (np.outer(c, c) * divided) @ q.T)
    value = result.norm_squared * float((c * c) @ values)
    return value, g


def value_and_grad(matvec: Callable[[Array], Array],
                   vjp: Callable[[Array, Array], Array], u: Array, depth: int,
                   f: Callable[[Array], Array], df: Callable[[Array], Array], *,
                   mode: str = "stored", block_size: int = 8,
                   reorthogonalize: bool = False) -> tuple[float, Array]:
    """Approximate a quadratic form and its parameter gradient.

    ``vjp(v, w)`` returns the parameter vector with entries ``w.T @ dA_j @ v``.
    It must hold v and w fixed. ``matvec`` and ``vjp`` must use the same fixed
    parameter point throughout a call. Replay requires a deterministic operator.
    Stored mode uses O(nm+m^2) workspace. Replay uses O(nb+m^2) workspace and
    up to ceil(m/b) additional recurrence passes (excluding callback workspace).
    """
    if mode not in ("stored", "replay"):
        raise ValueError("mode must be 'stored' or 'replay'")
    if mode == "replay" and reorthogonalize:
        raise ValueError("replay requires unreorthogonalized Lanczos")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    result = lanczos(matvec, u, depth, store_basis=mode == "stored",
                     reorthogonalize=reorthogonalize)
    value, g = projected_sensitivity(result, f, df)
    gradient = None

    def accumulate(v: Array, w: Array) -> None:
        nonlocal gradient
        contribution = np.asarray(vjp(v, w), dtype=np.float64)
        gradient = contribution.copy() if gradient is None else gradient + contribution

    m = len(result.alpha)
    if mode == "stored":
        w = result.basis @ g
        for j in range(m):
            accumulate(result.basis[:, j], w[:, j])
    else:
        for first in range(0, m, block_size):
            last = min(first + block_size, m)
            vb = np.empty((len(result.start), last - first))
            wb = np.zeros_like(vb)
            v = result.start.copy()
            previous = np.zeros_like(v)
            for i in range(m):
                wb += v[:, None] * g[i, first:last]
                if first <= i < last:
                    vb[:, i - first] = v
                if i < m - 1:
                    # Match the original recurrence's operation order.
                    z = np.asarray(matvec(v), dtype=np.float64)
                    z = z - (result.beta[i - 1] * previous if i else 0.0)
                    z = z - result.alpha[i] * v
                    previous, v = v, z / result.beta[i]
            for j in range(last - first):
                accumulate(vb[:, j], wb[:, j])
    return value, gradient
