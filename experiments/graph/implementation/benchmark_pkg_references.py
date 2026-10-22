from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, expm_multiply

from benchmark_pkg_directions import Direction

_HAVE_FAB = False
if os.path.isdir("fAb-frechet") and "fAb-frechet" not in sys.path:
    sys.path.append("fAb-frechet")
try:
    from lib import eval_fAb_krylov_sparse as ko_eval_fAb_krylov_sparse
    _HAVE_FAB = True
except Exception:
    _HAVE_FAB = False


@dataclass
class ReferenceEval:
    value: float
    grad: float
    runtime_sec: float
    kind: str


class _DirectionAsMatrix:
    """
    Lightweight wrapper so the Kressner-Oehme code can call E @ x
    without requiring us to materialize E explicitly.
    """
    def __init__(self, direction: Direction, n: int, scale: float = 1.0):
        self.direction = direction
        self.shape = (n, n)
        self.scale = scale

    def __matmul__(self, x):
        x = np.asarray(x).reshape(-1)
        return self.scale * self.direction.apply(x)

    def __mul__(self, alpha):
        return _DirectionAsMatrix(self.direction, self.shape[0], self.scale * float(alpha))

    def __rmul__(self, alpha):
        return _DirectionAsMatrix(self.direction, self.shape[0], self.scale * float(alpha))

    @property
    def T(self):
        # All directions we use here are symmetric.
        return self


def exact_dense_quadratic(A: sp.csr_matrix, direction: Direction, theta: float, u: np.ndarray, time_scale: float) -> ReferenceEval:
    E = direction.maybe_matrix()
    if E is None:
        raise ValueError("Dense exact reference requires an explicit direction matrix.")
    t0 = perf_counter()
    Ad = (A + theta * E).toarray()
    At = time_scale * Ad
    Ed = (time_scale * E).toarray()
    expA = la.expm(At)
    value = float(u @ expA @ u)
    L = la.expm_frechet(At, Ed, compute_expm=False)
    grad = float(u @ L @ u)
    return ReferenceEval(value=value, grad=grad, runtime_sec=perf_counter() - t0, kind="dense")


def _make_block_linear_operator(Atheta: sp.csr_matrix, direction: Direction, time_scale: float) -> LinearOperator:
    n = Atheta.shape[0]

    def matvec(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z).reshape(-1)
        x1 = z[:n]
        x2 = z[n:]
        top = time_scale * (Atheta @ x1) + time_scale * direction.apply(x2)
        bot = time_scale * (Atheta @ x2)
        return np.concatenate([top, bot])

    def rmatvec(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z).reshape(-1)
        y1 = z[:n]
        y2 = z[n:]
        top = time_scale * (Atheta.T @ y1)
        bot = time_scale * direction.apply(y1) + time_scale * (Atheta.T @ y2)
        return np.concatenate([top, bot])

    def matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.column_stack([matvec(X[:, j]) for j in range(X.shape[1])])

    def rmatmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.column_stack([rmatvec(X[:, j]) for j in range(X.shape[1])])

    return LinearOperator((2 * n, 2 * n), matvec=matvec, rmatvec=rmatvec, matmat=matmat, rmatmat=rmatmat, dtype=float)


def sparse_block_quadratic(A: sp.csr_matrix, direction: Direction, theta: float, u: np.ndarray, time_scale: float) -> ReferenceEval:
    E = direction.maybe_matrix()
    Atheta = A.copy() if E is None else (A + theta * E).tocsr()

    t0 = perf_counter()

    At = (time_scale * Atheta).tocsr()
    traceA = float(At.diagonal().sum())
    expAu = expm_multiply(At, u, traceA=traceA)
    value = float(u @ expAu)

    if E is not None:
        Et = (time_scale * E).tocsr()
        n = Atheta.shape[0]
        Z = sp.csr_matrix((n, n))
        B = sp.bmat([[At, Et], [Z, At]], format="csr")
        rhs = np.concatenate([np.zeros(n), u])
        out = expm_multiply(B, rhs, traceA=2.0 * traceA)
    else:
        B = _make_block_linear_operator(Atheta, direction, time_scale)
        rhs = np.concatenate([np.zeros(Atheta.shape[0]), u])
        out = expm_multiply(B, rhs, traceA=2.0 * traceA)

    grad = float(u @ out[: Atheta.shape[0]])
    return ReferenceEval(value=value, grad=grad, runtime_sec=perf_counter() - t0, kind="sparse_block")


def ko_quadratic(A: sp.csr_matrix, direction: Direction, theta: float, u: np.ndarray, k: int, time_scale: float) -> ReferenceEval:
    if not _HAVE_FAB:
        raise RuntimeError("Could not import fAb-frechet/lib.py")

    E_explicit = direction.maybe_matrix()
    Atheta = A if E_explicit is None else (A + theta * E_explicit).tocsr()

    if E_explicit is not None:
        E_for_ko = time_scale * E_explicit
    else:
        E_for_ko = _DirectionAsMatrix(direction, A.shape[0], scale=time_scale)

    t0 = perf_counter()
    At = (time_scale * Atheta).tocsr()
    fAb, dfAb = ko_eval_fAb_krylov_sparse(la.expm, At, u, E_for_ko, k)
    value = float(u @ fAb)
    grad = float(u @ dfAb)
    return ReferenceEval(value=value, grad=grad, runtime_sec=perf_counter() - t0, kind="modified_arnoldi")


def ko_bilinear(
    A: sp.csr_matrix,
    direction: Direction,
    theta: float,
    u: np.ndarray,
    v: np.ndarray,
    k: int,
    time_scale: float,
) -> ReferenceEval:
    """
    Direct Modified-Arnoldi bilinear form:
        u^T exp(tA) v
        u^T L_exp(tA, tE) v

    Do NOT use polarization here. For directions like E=11^T,
    polarization can create starting vectors annihilated by E and trigger
    breakdown in their separate-orthogonalization code.
    """
    if not _HAVE_FAB:
        raise RuntimeError("Could not import fAb-frechet/lib.py")

    E_explicit = direction.maybe_matrix()
    Atheta = A if E_explicit is None else (A + theta * E_explicit).tocsr()

    if E_explicit is not None:
        E_for_ko = time_scale * E_explicit
    else:
        E_for_ko = _DirectionAsMatrix(direction, A.shape[0], scale=time_scale)

    t0 = perf_counter()
    At = (time_scale * Atheta).tocsr()

    fAv, dfAv = ko_eval_fAb_krylov_sparse(la.expm, At, v, E_for_ko, k)

    value = float(u @ fAv)
    grad = float(u @ dfAv)

    return ReferenceEval(
        value=value,
        grad=grad,
        runtime_sec=perf_counter() - t0,
        kind="modified_arnoldi",
    )