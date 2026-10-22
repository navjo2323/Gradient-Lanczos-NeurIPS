from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, expm, expm_frechet


Array = np.ndarray

I2 = np.eye(2, dtype=np.float64)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def kron_all(ops: Sequence[Array]) -> Array:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_tfim_basis(num_sites: int) -> Tuple[List[Array], List[str]]:
    basis: List[Array] = []
    names: List[str] = []
    for i in range(num_sites):
        ops = [I2] * num_sites
        ops[i] = X
        basis.append(kron_all(ops))
        names.append(f"X_{i}")
    for i in range(num_sites):
        ops = [I2] * num_sites
        ops[i] = Z
        basis.append(kron_all(ops))
        names.append(f"Z_{i}")
    for i in range(num_sites - 1):
        ops = [I2] * num_sites
        ops[i] = Z
        ops[i + 1] = Z
        basis.append(kron_all(ops))
        names.append(f"ZZ_{i}_{i+1}")
    return basis, names


def assemble_h(theta: Array, basis: Sequence[Array]) -> Array:
    H = np.zeros_like(basis[0])
    for t, P in zip(theta, basis):
        H += t * P
    return 0.5 * (H + H.T)


@dataclass
class Dataset:
    xs: List[np.ndarray]
    cs: List[np.ndarray]
    ts: List[float]
    theta_true: np.ndarray
    basis_names: List[str]


def make_tfim_theta(
    num_sites: int,
    rng: np.random.Generator,
    disorder: float,
    regime: str,
) -> np.ndarray:
    p = 3 * num_sites - 1
    theta = np.zeros(p, dtype=np.float64)

    if regime == "easy":
        hx_base = 0.35
        hz_base = 0.40
        J_base = 1.0
    elif regime == "hard":
        hx_base = 1.0
        hz_base = 0.05
        J_base = 1.0
    else:
        raise ValueError(f"Unknown regime: {regime}")

    theta[:num_sites] = hx_base + disorder * rng.normal(size=num_sites)
    theta[num_sites:2 * num_sites] = hz_base + disorder * rng.normal(size=num_sites)
    theta[2 * num_sites :] = J_base + disorder * rng.normal(size=num_sites - 1)
    return theta


def random_real_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)
    return x


def make_dataset(
    num_sites: int,
    samples: int,
    t_min: float,
    t_max: float,
    seed: int,
    disorder: float,
    regime: str,
) -> tuple[List[np.ndarray], List[str], Dataset]:
    rng = np.random.default_rng(seed)
    basis, basis_names = build_tfim_basis(num_sites)
    theta_true = make_tfim_theta(num_sites, rng, disorder=disorder, regime=regime)
    H_true = assemble_h(theta_true, basis)
    dim = H_true.shape[0]

    xs: List[np.ndarray] = []
    cs: List[np.ndarray] = []
    ts: List[float] = []

    for _ in range(samples):
        x = random_real_state(dim, rng).astype(np.complex128)
        t = float(rng.uniform(t_min, t_max))
        U = expm(-1j * t * H_true)
        c = U @ x
        xs.append(x)
        cs.append(c)
        ts.append(t)

    return basis, basis_names, Dataset(xs=xs, cs=cs, ts=ts, theta_true=theta_true, basis_names=basis_names)


# ---------- Dense exact gradient ----------

def dense_loss_and_grad(theta: np.ndarray, basis: Sequence[Array], data: Dataset) -> tuple[float, np.ndarray]:
    H = assemble_h(theta, basis)
    grad = np.zeros_like(theta)
    loss = 0.0
    for x, c, t in zip(data.xs, data.cs, data.ts):
        A = -1j * t * H
        U = expm(A)
        r = c - U @ x
        loss += float(np.vdot(r, r).real)
        for j, Pj in enumerate(basis):
            E = -1j * t * Pj
            dU = expm_frechet(A, E, compute_expm=False)
            grad[j] += -2.0 * np.real(np.vdot(r, dU @ x))
    return loss, grad


# ---------- Lanczos/Gm for real-symmetric matrix functions ----------

def lanczos(H: Array, v: Array, m: int) -> tuple[List[Array], np.ndarray, np.ndarray, np.ndarray]:
    n = H.shape[0]
    x_prev = np.zeros(n, dtype=np.float64)
    x_cur = np.array(v, dtype=np.float64, copy=True)
    x_cur /= np.linalg.norm(x_cur)
    xs = [x_cur.copy()]
    alphas = np.zeros(m, dtype=np.float64)
    betas = np.zeros(m, dtype=np.float64)
    b_prev = 1.0
    for k in range(m):
        w = H @ x_cur - b_prev * x_prev
        a = float(x_cur @ w)
        w = w - a * x_cur
        b = float(np.linalg.norm(w))
        alphas[k] = a
        betas[k] = b
        if b < 1e-14:
            x_next = np.zeros_like(x_cur)
        else:
            x_next = w / b
        xs.append(x_next.copy())
        x_prev = x_cur
        x_cur = x_next
        b_prev = b
    T = np.diag(alphas)
    for k in range(m - 1):
        T[k, k + 1] = betas[k]
        T[k + 1, k] = betas[k]
    return xs, alphas, betas, T


def divided_diffs(evals: Array, f: Callable[[Array], Array], fp: Callable[[Array], Array]) -> Array:
    m = len(evals)
    fvals = f(evals)
    F = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                F[i, j] = float(fp(np.array([evals[i]]))[0])
            else:
                d = evals[i] - evals[j]
                F[i, j] = float((fvals[i] - fvals[j]) / d) if abs(d) > 1e-14 else float(fp(np.array([evals[i]]))[0])
    return F


def small_G(T: Array, rho_v: float, f: Callable[[Array], Array], fp: Callable[[Array], Array]) -> Array:
    evals, Q = eigh(T)
    e1 = np.zeros(T.shape[0])
    e1[0] = 1.0
    c = Q.T @ e1
    F = divided_diffs(evals, f, fp)
    return Q @ (rho_v * (np.outer(c, c) * F)) @ Q.T


def quad_grad_realvec(H: Array, basis: Sequence[Array], v: Array, m: int, f, fp) -> np.ndarray:
    rho_v = float(v @ v)
    if rho_v < 1e-15:
        return np.zeros(len(basis))
    xs, _, _, T = lanczos(H, v, m)
    V = np.column_stack(xs[:-1])
    G = small_G(T, rho_v, f, fp)
    M = V @ G @ V.T
    return np.array([float(np.sum(M * P)) for P in basis])


def quad_grad_complexvec(H: Array, basis: Sequence[Array], z: np.ndarray, m: int, f, fp) -> np.ndarray:
    a = np.real(z)
    b = np.imag(z)
    return quad_grad_realvec(H, basis, a, m, f, fp) + quad_grad_realvec(H, basis, b, m, f, fp)


def polarization_terms(u: np.ndarray, v: np.ndarray) -> List[tuple[complex, np.ndarray]]:
    out = []
    for ell in range(4):
        w = 1j ** ell
        out.append((0.25 * np.conjugate(w), u + w * v))
    return out


def gm_loss_and_grad(theta: np.ndarray, basis: Sequence[Array], data: Dataset, m: int) -> tuple[float, np.ndarray]:
    H = assemble_h(theta, basis)
    loss = 0.0
    grad = np.zeros_like(theta)
    for x, c, t in zip(data.xs, data.cs, data.ts):
        U = expm(-1j * t * H)
        r = c - U @ x
        loss += float(np.vdot(r, r).real)

        fC = lambda lam: np.cos(t * lam)
        fpC = lambda lam: -t * np.sin(t * lam)
        fS = lambda lam: np.sin(t * lam)
        fpS = lambda lam: t * np.cos(t * lam)

        grad_BC = np.zeros_like(theta, dtype=np.complex128)
        grad_BS = np.zeros_like(theta, dtype=np.complex128)
        for weight, z in polarization_terms(c, x):
            grad_BC += weight * quad_grad_complexvec(H, basis, z, m, fC, fpC)
            grad_BS += weight * quad_grad_complexvec(H, basis, z, m, fS, fpS)

        grad += -2.0 * (np.real(grad_BC) + np.imag(grad_BS))
    return loss, grad


# ---------- Diagnostics ----------

def grad_diagnostics(theta: np.ndarray, basis: Sequence[Array], data: Dataset, depth: int) -> dict:
    loss_dense, g_dense = dense_loss_and_grad(theta, basis, data)
    loss_gm, g_gm = gm_loss_and_grad(theta, basis, data, depth)

    nd = np.linalg.norm(g_dense)
    ng = np.linalg.norm(g_gm)
    rel = np.linalg.norm(g_gm - g_dense) / max(nd, 1e-14)
    cos = float(np.dot(g_dense, g_gm) / max(nd * ng, 1e-14))

    return {
        "loss_dense": float(loss_dense),
        "loss_gm": float(loss_gm),
        "grad_norm_dense": float(nd),
        "grad_norm_gm": float(ng),
        "grad_rel_error": float(rel),
        "grad_cosine": float(cos),
    }


def sweep_gradient_diagnostics(theta: np.ndarray, basis: Sequence[Array], data: Dataset, depths: Sequence[int]):
    rows = []
    _, g_dense = dense_loss_and_grad(theta, basis, data)
    nd = np.linalg.norm(g_dense)

    print("\nGradient diagnostics at initialization")
    print("depth | rel_error | cosine | ||g_dense|| | ||g_gm||")

    for depth in depths:
        _, g_gm = gm_loss_and_grad(theta, basis, data, depth)
        ng = np.linalg.norm(g_gm)
        rel = np.linalg.norm(g_gm - g_dense) / max(nd, 1e-14)
        cos = float(np.dot(g_dense, g_gm) / max(nd * ng, 1e-14))
        rows.append(
            {
                "depth": int(depth),
                "grad_rel_error": float(rel),
                "grad_cosine": float(cos),
                "grad_norm_dense": float(nd),
                "grad_norm_gm": float(ng),
            }
        )
        print(f"{depth:5d} | {rel:9.3e} | {cos:6.4f} | {nd:10.3e} | {ng:10.3e}")

    return rows


# ---------- Optimizer ----------

def adam_step(theta, grad, m, v, lr, beta1, beta2, eps, t):
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    mhat = m / (1.0 - beta1 ** t)
    vhat = v / (1.0 - beta2 ** t)
    theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
    return theta, m, v


def train(method: str, theta0: np.ndarray, basis: Sequence[Array], data: Dataset, depth: int | None, steps: int, lr: float):
    theta = theta0.copy()
    m1 = np.zeros_like(theta)
    m2 = np.zeros_like(theta)
    hist = []
    total_runtime = 0.0
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        if method == "dense":
            loss, grad = dense_loss_and_grad(theta, basis, data)
        elif method == "gm":
            if depth is None:
                raise ValueError("depth required for gm")
            loss, grad = gm_loss_and_grad(theta, basis, data, depth)
        else:
            raise ValueError(method)
        total_runtime += time.perf_counter() - t0
        theta, m1, m2 = adam_step(theta, grad, m1, m2, lr, 0.9, 0.999, 1e-8, step)
        hist.append({
            "step": step,
            "loss": float(loss),
            "param_error": float(np.linalg.norm(theta - data.theta_true)),
            "grad_norm": float(np.linalg.norm(grad)),
            "wall_time": float(total_runtime),
        })
    return theta, hist


def plot_histories(out_dir: str, dense_hist, gm_hists):
    os.makedirs(out_dir, exist_ok=True)
    for key, ylabel, fname, logy in [
        ("loss", "Loss", "loss_vs_step.png", True),
        ("param_error", "Parameter error", "param_error_vs_step.png", True),
        ("loss", "Loss", "loss_vs_time.png", True),
        ("param_error", "Parameter error", "param_error_vs_time.png", True),
    ]:
        plt.figure(figsize=(7, 4.5))
        if "time" in fname:
            xd = [h["wall_time"] for h in dense_hist]
            xlabel = "Wall time (sec)"
        else:
            xd = [h["step"] for h in dense_hist]
            xlabel = "Step"

        yd = [h[key] for h in dense_hist]
        plt.plot(xd, yd, label="dense", linewidth=2)

        for depth, hist in gm_hists.items():
            if "time" in fname:
                xg = [h["wall_time"] for h in hist]
            else:
                xg = [h["step"] for h in hist]
            yg = [h[key] for h in hist]
            plt.plot(xg, yg, label=f"gm m={depth}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if logy:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--depths", type=int, nargs="+", default=[20, 30, 40])
    ap.add_argument("--diag-depths", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--t-min", type=float, default=2.0)
    ap.add_argument("--t-max", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disorder", type=float, default=0.05)
    ap.add_argument("--regime", choices=["easy", "hard"], default="easy")
    ap.add_argument("--init-scale", type=float, default=0.1)
    ap.add_argument("--out-dir", type=str, default="gm_vs_dense_outputs")
    args = ap.parse_args()

    basis, basis_names, data = make_dataset(
        num_sites=args.sites,
        samples=args.samples,
        t_min=args.t_min,
        t_max=args.t_max,
        seed=args.seed,
        disorder=args.disorder,
        regime=args.regime,
    )

    rng = np.random.default_rng(args.seed + 999)
    theta0 = data.theta_true + args.init_scale * rng.normal(size=len(data.theta_true))

    dim = 2 ** args.sites
    depths = sorted(set([d for d in args.depths if d < dim]))
    if not depths:
        raise ValueError(f"All requested depths are >= dimension={dim}")

    diag_depths = (
        args.diag_depths
        if args.diag_depths is not None and len(args.diag_depths) > 0
        else sorted(set([max(5, depths[0] // 2), *depths]))
    )
    diag_depths = sorted(set([d for d in diag_depths if d < dim]))

    print("\nDataset / problem summary")
    print(f"sites: {args.sites}")
    print(f"dimension: {dim}")
    print(f"samples: {args.samples}")
    print(f"regime: {args.regime}")
    print(f"disorder: {args.disorder}")
    print(f"time range: [{args.t_min}, {args.t_max}]")
    print(f"depths: {depths}")
    print(f"init scale: {args.init_scale}")

    dense_histories = {}
    gm_histories = {}

    grad_sweep = sweep_gradient_diagnostics(theta0, basis, data, diag_depths)

    theta_dense, hist_dense = train("dense", theta0, basis, data, None, args.steps, args.lr)

    gm_results = {}
    for depth in depths:
        print(f"\nTraining GM with depth {depth}")
        grad_diag = grad_diagnostics(theta0, basis, data, depth)
        for k, v in grad_diag.items():
            print(f"{k}: {v}")
        theta_gm, hist_gm = train("gm", theta0, basis, data, depth, args.steps, args.lr)
        gm_results[str(depth)] = {
            "theta": theta_gm.tolist(),
            "history": hist_gm,
            "initial_gradient_diagnostic": grad_diag,
        }
        gm_histories[depth] = hist_gm

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "basis_names": basis_names,
                "theta_true": data.theta_true.tolist(),
                "theta0": theta0.tolist(),
                "theta_dense": theta_dense.tolist(),
                "dense_history": hist_dense,
                "initial_gradient_sweep": grad_sweep,
                "gm_runs": gm_results,
            },
            f,
            indent=2,
        )

    plot_histories(args.out_dir, hist_dense, gm_histories)

    print(f"\ndense final param error: {np.linalg.norm(theta_dense - data.theta_true):.3e}")
    for depth in depths:
        theta_gm = np.array(gm_results[str(depth)]["theta"])
        print(f"gm m={depth:3d} final param error: {np.linalg.norm(theta_gm - data.theta_true):.3e}")

    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()