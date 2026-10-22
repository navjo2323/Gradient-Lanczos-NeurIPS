from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp


class Direction:
    name: str

    def contraction(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def apply(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def projected_from_basis(self, basis: list[np.ndarray], k_eff: int) -> np.ndarray:
        """
        Return V^T E V using the stored Lanczos basis vectors x_1,...,x_k,
        without materializing the dense n x k matrix V when possible.
        """
        raise NotImplementedError

    def maybe_matrix(self) -> Optional[sp.csr_matrix]:
        return None


@dataclass
class EdgeDirection(Direction):
    n: int
    i: int
    j: int
    symmetric: bool = True
    name: str = "edge"

    def contraction(self, x: np.ndarray, y: np.ndarray) -> float:
        out = x[self.i] * y[self.j]
        if self.symmetric and self.i != self.j:
            out += x[self.j] * y[self.i]
        return float(out)

    def apply(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        out[self.i] += y[self.j]
        if self.symmetric and self.i != self.j:
            out[self.j] += y[self.i]
        return out

    def projected_from_basis(self, basis: list[np.ndarray], k_eff: int) -> np.ndarray:
        ri = np.fromiter((basis[t][self.i] for t in range(k_eff)), dtype=float, count=k_eff)
        rj = np.fromiter((basis[t][self.j] for t in range(k_eff)), dtype=float, count=k_eff)
        out = np.outer(ri, rj)
        if self.symmetric and self.i != self.j:
            out = out + np.outer(rj, ri)
        return out

    def maybe_matrix(self) -> sp.csr_matrix:
        rows = [self.i]
        cols = [self.j]
        vals = [1.0]
        if self.symmetric and self.i != self.j:
            rows.append(self.j)
            cols.append(self.i)
            vals.append(1.0)
        return sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()


@dataclass
class OnesOuterDirection(Direction):
    n: int
    name: str = "ones_outer"

    def contraction(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(x) * np.sum(y))

    def apply(self, y: np.ndarray) -> np.ndarray:
        return np.full(self.n, np.sum(y), dtype=float)

    def projected_from_basis(self, basis: list[np.ndarray], k_eff: int) -> np.ndarray:
        s = np.fromiter((float(np.sum(basis[t])) for t in range(k_eff)), dtype=float, count=k_eff)
        return np.outer(s, s)


@dataclass
class NodeOuterDirection(Direction):
    n: int
    ell: int
    name: str = "node_outer"

    def contraction(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(x[self.ell] * y[self.ell])

    def apply(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        out[self.ell] = y[self.ell]
        return out

    def projected_from_basis(self, basis: list[np.ndarray], k_eff: int) -> np.ndarray:
        r = np.fromiter((basis[t][self.ell] for t in range(k_eff)), dtype=float, count=k_eff)
        return np.outer(r, r)

    def maybe_matrix(self) -> sp.csr_matrix:
        return sp.coo_matrix(([1.0], ([self.ell], [self.ell])), shape=(self.n, self.n)).tocsr()


@dataclass
class IdentityDirection(Direction):
    n: int
    name: str = "identity"

    def contraction(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(x, y))

    def apply(self, y: np.ndarray) -> np.ndarray:
        return y.copy()

    def projected_from_basis(self, basis: list[np.ndarray], k_eff: int) -> np.ndarray:
        # Keep the exact Gram computation instead of assuming orthonormality.
        G = np.empty((k_eff, k_eff), dtype=float)
        for i in range(k_eff):
            xi = basis[i]
            for j in range(i, k_eff):
                gij = float(np.dot(xi, basis[j]))
                G[i, j] = gij
                G[j, i] = gij
        return G

    def maybe_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n, format="csr")