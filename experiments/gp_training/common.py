"""Paper-only helpers extracted from the original GP experiment driver.

Numerical data generation, preprocessing, and subset selection are unchanged.
Only dataset acquisition is replaced with a pinned, hash-checked local cache.
"""
from __future__ import annotations
import dataclasses
import os
import tempfile
from pathlib import Path

_keops_cache = Path(tempfile.gettempdir()) / "lanczos_keops_cache"
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(_keops_cache))
os.environ.setdefault("PYKEOPS_CACHE_FOLDER", str(_keops_cache))
_keops_cache.mkdir(parents=True, exist_ok=True)
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KeOpsMaternKernel
from linear_operator.operators import DiagLinearOperator
from linear_operator.utils.lanczos import lanczos_tridiag
from data import load_protein
Array = jax.Array
PAPER_GP_DATASETS = ("protein",)
DATASET_ALIASES = {"toy": "toy", "protein": "protein"}


def _load_uci_dataset(name):
    if name != "protein":
        raise ValueError("This release supplies the Protein experiment; use --dataset toy for a quick run.")
    raw_x, raw_y = load_protein()
    x, y = _standardize_regression_data(raw_x, raw_y)
    return x, y, {"dataset": "protein", "source": "hash-verified uci_datasets protein/data.csv.gz", "full_n": int(x.shape[0]), "full_d": int(x.shape[1]), "is_paper_gp_dataset": True, "paper_size": 45730, "paper_dim": 9}

@dataclasses.dataclass
class Config:
    n: int = 2048
    d: int = 2
    lanczos_depth: int = 20
    num_probes: int = 64
    seed: int = 0
    jitter: float = 1e-6
    repeats: int = 5
    save_json: bool = True
    torch_dtype: str = "float64"
    debug_backends: bool = False
    disable_their_adjoints: bool = False
    torch_kernel_backend: str = "standard"  # "standard" or "keops"
    ours_grad_backend: str = "jax"  # "jax" or "torch_keops"
    dataset: str = "toy"
    all_data: bool = False
    kernel: str = "rbf"  # "rbf" or "matern32"
    exact_mode: str = "auto"  # "auto", "dense", or "none"
    exact_backend: str = "jax_dense"  # "jax_dense" or "torch_cholesky"
    dense_exact_max_n: int = 4096

def make_toy_regression(key: Array, n: int, d: int) -> tuple[Array, Array]:
    key_x, key_w, key_eps = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (n, d), dtype=jnp.float64)
    w = jax.random.normal(key_w, (d,), dtype=jnp.float64)
    y_clean = jnp.sin(x @ w) + 0.25 * jnp.cos(2.0 * x[:, 0])
    y = y_clean + 0.05 * jax.random.normal(key_eps, (n,), dtype=jnp.float64)
    return x, y

def _canonical_dataset_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        expected = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset {name!r}. Expected one of: {expected}")
    return DATASET_ALIASES[key]

def _standardize_regression_data(inputs: Any, targets: Any) -> tuple[Array, Array]:
    x_np = np.asarray(inputs, dtype=np.float64)
    y_np = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x_np.ndim != 2:
        raise ValueError(f"Expected inputs with shape (n,d), got {x_np.shape}")
    if y_np.shape[0] != x_np.shape[0]:
        raise ValueError(f"Inputs/targets disagree: {x_np.shape} versus {y_np.shape}")

    col_std = np.nanstd(x_np, axis=0)
    keep = np.isfinite(col_std) & (col_std > 1e-12)
    x_np = x_np[:, keep]
    mean = np.nanmean(x_np, axis=0, keepdims=True)
    std = np.nanstd(x_np, axis=0, keepdims=True) + 1e-6
    x_np = (x_np - mean) / std

    y_np = y_np - np.nanmean(y_np)
    y_std = np.nanstd(y_np)
    if np.isfinite(y_std) and y_std > 0.0:
        y_np = y_np / y_std
    return jnp.asarray(x_np, dtype=jnp.float64), jnp.asarray(y_np, dtype=jnp.float64)

def make_regression_data(config: Config) -> tuple[Array, Array, dict[str, Any]]:
    dataset = _canonical_dataset_name(config.dataset)
    key = jax.random.PRNGKey(config.seed)
    if dataset == "toy":
        x, y = make_toy_regression(key, config.n, config.d)
        metadata = {
            "dataset": "toy",
            "source": "synthetic sinusoid/cosine regression",
            "full_n": int(x.shape[0]),
            "full_d": int(x.shape[1]),
            "selected_n": int(x.shape[0]),
            "selected_d": int(x.shape[1]),
            "selection": "generated",
            "is_paper_gp_dataset": False,
        }
        return x, y, metadata

    x_full, y_full, metadata = _load_uci_dataset(dataset)
    full_n = int(x_full.shape[0])
    if config.all_data or config.n <= 0:
        n_selected = full_n
        selection = "all"
        idx = np.arange(full_n)
    else:
        n_selected = min(int(config.n), full_n)
        rng = np.random.default_rng(int(config.seed))
        idx = rng.choice(full_n, size=n_selected, replace=False)
        idx.sort()
        selection = f"seeded_subset_seed_{config.seed}"

    x = x_full[idx]
    y = y_full[idx]
    metadata = {
        **metadata,
        "selected_n": int(x.shape[0]),
        "selected_d": int(x.shape[1]),
        "selection": selection,
        "paper_gp_defaults": {
            "kernel": "matern32",
            "lanczos_depth": 10,
            "num_probes": 10,
            "datasets": list(PAPER_GP_DATASETS),
        } if dataset in PAPER_GP_DATASETS else None,
    }
    return x, y, metadata

def _reset_gpytorch_probe_cache() -> None:
    """Clear deterministic-probe caches so each run/depth uses the requested n,p."""
    for owner in []:
        pass
    try:
        gpytorch.settings.deterministic_probes.probe_vectors = None
    except Exception:
        pass
    try:
        import linear_operator
        linear_operator.settings.deterministic_probes.probe_vectors = None
    except Exception:
        pass

def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {name}")
