from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_graph_matrix(path: str, symmetrize: bool = True, remove_self_loops: bool = True) -> sp.csr_matrix:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".mtx":
        A = scipy.io.mmread(path).tocsr()
    elif suffix == ".npz":
        A = sp.load_npz(path).tocsr()
    else:
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] == 2:
            rows = data[:, 0].astype(int)
            cols = data[:, 1].astype(int)
            vals = np.ones(len(rows), dtype=float)
        elif data.shape[1] >= 3:
            rows = data[:, 0].astype(int)
            cols = data[:, 1].astype(int)
            vals = data[:, 2].astype(float)
        else:
            raise ValueError("Expected edge list with 2 or 3 columns.")

        offset = 1 if min(rows.min(), cols.min()) == 1 else 0
        rows = rows - offset
        cols = cols - offset

        nodes = np.unique(np.concatenate([rows, cols]))
        remap = {node: k for k, node in enumerate(nodes)}
        rows = np.array([remap[r] for r in rows], dtype=int)
        cols = np.array([remap[c] for c in cols], dtype=int)
        A = sp.coo_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()

    A = A.astype(float).tocsr()

    if remove_self_loops:
        A = A.tolil()
        A.setdiag(0.0)
        A = A.tocsr()
        A.eliminate_zeros()

    if symmetrize:
        A = 0.5 * (A + A.T)
        A = A.tocsr()
        A.eliminate_zeros()

    return A


def induced_subgraph(A: sp.csr_matrix, m: int) -> sp.csr_matrix:
    if not (1 <= m <= A.shape[0]):
        raise ValueError("subgraph size must be between 1 and n")
    idx = np.arange(m)
    return A[idx][:, idx].tocsr()


def choose_existing_edge(A: sp.csr_matrix) -> tuple[int, int]:
    upper = sp.triu(A, k=1).tocoo()
    if upper.nnz == 0:
        raise ValueError("No off-diagonal edge found.")
    return int(upper.row[0]), int(upper.col[0])


def unit_vector(n: int, i: int) -> np.ndarray:
    e = np.zeros(n, dtype=float)
    e[i] = 1.0
    return e


def apply_matrix_transform(A: sp.csr_matrix, matrix_kind: str) -> sp.csr_matrix:
    """
    Supported transforms:
      - raw_adj: A
      - norm_adj: D^{-1/2} A D^{-1/2}
      - neg_norm_laplacian: D^{-1/2} A D^{-1/2} - I = -L_norm
    """
    matrix_kind = matrix_kind.lower()
    if matrix_kind == "raw_adj":
        return A.tocsr()

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    inv_sqrt_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_sqrt_deg[mask] = 1.0 / np.sqrt(deg[mask])

    D_inv_sqrt = sp.diags(inv_sqrt_deg, offsets=0, format="csr")
    S = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

    if matrix_kind == "norm_adj":
        return S

    if matrix_kind == "neg_norm_laplacian":
        n = A.shape[0]
        return (S - sp.eye(n, format="csr")).tocsr()

    raise ValueError(f"Unknown matrix_kind: {matrix_kind}")