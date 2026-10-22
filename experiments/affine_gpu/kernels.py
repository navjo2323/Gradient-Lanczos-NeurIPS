"""Verbatim shared definitions used by the paper affine experiment.

Includes transitive helpers from the original recurrence; the release runner
selects only the paper's Hadamard operator and full/no reorthogonalization.
See source_manifest.json for extraction provenance.
"""
from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class LanczosStats:
    steps_done: int = 0
    event_count: int = 0
    selected_fraction_sum: float = 0.0
    max_gram_offdiag: float = float("nan")
    orthogonalization_seconds: float = 0.0
    matvec_seconds: float = 0.0
    pro_rows: list[dict[str, float | int | bool | str]] | None = None

    def record_event(self, selected_count: int, valid_count: int) -> None:
        self.event_count += 1
        if valid_count > 0:
            self.selected_fraction_sum += selected_count / float(valid_count)

    def event_fraction(self) -> float:
        return self.event_count / float(max(self.steps_done, 1))

    def mean_selected_fraction(self) -> float:
        if self.event_count == 0:
            return 0.0
        return self.selected_fraction_sum / float(self.event_count)



def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)



def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)



def make_cluster_spectrum(
    n: int,
    clusters: int,
    condition: float,
    width: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    centers = torch.logspace(-math.log10(condition), 0.0, clusters, dtype=dtype)
    repeats = math.ceil(n / clusters)
    spectrum = centers.repeat_interleave(repeats)[:n].clone()
    if width > 0.0:
        jitter = width * torch.randn(n, dtype=dtype, generator=generator)
        spectrum = spectrum * torch.exp(jitter)
    order = torch.randperm(n, generator=generator)
    return spectrum[order].to(device)



def fwht(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError("fwht expects a vector")
    n = vector.numel()
    y = vector
    h = 1
    while h < n:
        y = y.reshape(-1, 2 * h)
        left = y[:, :h]
        right = y[:, h:]
        y = torch.cat((left + right, left - right), dim=1)
        h *= 2
    return y.reshape(n)



def laplacian_tiles_base(vector: torch.Tensor, side: int, tile_size: int, boundary_weight: float) -> torch.Tensor:
    grid = vector.reshape(side, side)
    result = torch.zeros_like(grid)
    rows = torch.arange(side, device=vector.device).reshape(side, 1)
    cols = torch.arange(side, device=vector.device).reshape(1, side)
    horizontal_intra = ((cols + 1) < side) & (((cols + 1) % tile_size) != 0)
    vertical_intra = ((rows + 1) < side) & (((rows + 1) % tile_size) != 0)
    horizontal_bridge = ((cols + 1) < side) & (((cols + 1) % tile_size) == 0)
    vertical_bridge = ((rows + 1) < side) & (((rows + 1) % tile_size) == 0)
    for dim, active, weight in (
        (1, horizontal_intra, 1.0),
        (0, vertical_intra, 1.0),
        (1, horizontal_bridge, boundary_weight),
        (0, vertical_bridge, boundary_weight),
    ):
        neighbor = torch.roll(grid, shifts=-1, dims=dim)
        diff = grid - neighbor
        contribution = torch.where(active, diff, torch.zeros_like(diff))
        result = result + weight * contribution
        result = result + weight * torch.roll(torch.where(active, -diff, torch.zeros_like(diff)), shifts=1, dims=dim)
    return result.reshape(-1)



def planted_partition_laplacian(
    n: int,
    communities: int,
    bridge_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n % communities != 0:
        raise ValueError("--n must be divisible by --communities for planted-partition")
    block_size = n // communities
    adjacency = torch.zeros((n, n), dtype=dtype, device=device)
    for community in range(communities):
        start = community * block_size
        stop = start + block_size
        adjacency[start:stop, start:stop] = 1.0
        adjacency[start:stop, start:stop].fill_diagonal_(0.0)
    for community in range(communities):
        left = community * block_size
        right = ((community + 1) % communities) * block_size
        adjacency[left, right] = bridge_weight
        adjacency[right, left] = bridge_weight
    degree = torch.sum(adjacency, dim=1)
    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=1.0e-30))
    normalized_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return torch.eye(n, dtype=dtype, device=device) - normalized_adjacency



def sbm_laplacian(
    n: int,
    communities: int,
    p_in: float,
    p_out: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n % communities != 0:
        raise ValueError("--n must be divisible by --communities for sbm-laplacian")
    block_size = n // communities
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    labels = torch.arange(n, device="cpu") // block_size
    same = labels[:, None] == labels[None, :]
    probabilities = torch.where(
        same,
        torch.full((n, n), p_in, dtype=torch.float32),
        torch.full((n, n), p_out, dtype=torch.float32),
    )
    upper = torch.rand((n, n), generator=generator) < probabilities
    upper = torch.triu(upper, diagonal=1)
    adjacency = (upper | upper.T).to(dtype=dtype, device=device)
    degree = torch.sum(adjacency, dim=1)
    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=1.0e-30))
    normalized_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return torch.eye(n, dtype=dtype, device=device) - normalized_adjacency



def sparse_sbm_edges(
    n: int,
    communities: int,
    p_in: float,
    p_out: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n % communities != 0:
        raise ValueError("--n must be divisible by --communities for sbm-sparse")
    block_size = n // communities
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []

    for community in range(communities):
        start = community * block_size
        upper = torch.rand((block_size, block_size), generator=generator) < p_in
        rows, cols = torch.nonzero(torch.triu(upper, diagonal=1), as_tuple=True)
        if rows.numel() == 0:
            continue
        rows = rows + start
        cols = cols + start
        src_parts.extend([rows, cols])
        dst_parts.extend([cols, rows])

    inter_degree = max(1, int(round(p_out * n)))
    nodes = torch.arange(n, dtype=torch.long)
    labels = nodes // block_size
    inter_src = nodes.repeat_interleave(inter_degree)
    inter_dst = torch.randint(0, n, (n * inter_degree,), generator=generator)
    same = (inter_dst // block_size) == labels.repeat_interleave(inter_degree)
    if torch.any(same):
        inter_dst[same] = (inter_dst[same] + block_size) % n
    src_parts.extend([inter_src, inter_dst])
    dst_parts.extend([inter_dst, inter_src])

    src = torch.cat(src_parts).to(dtype=torch.long)
    dst = torch.cat(dst_parts).to(dtype=torch.long)
    order = src * n + dst
    unique = torch.unique(order)
    src = torch.div(unique, n, rounding_mode="floor")
    dst = unique - src * n

    degree = torch.bincount(src, minlength=n).to(dtype=torch.float64)
    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=1.0e-30))
    weight = (inv_sqrt_degree[src] * inv_sqrt_degree[dst]).to(dtype=dtype)
    return src.to(device), dst.to(device), weight.to(device)



class Operator:
    def __init__(self, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
        self.args = args
        self.device = device
        self.dtype = dtype
        self.n = args.n
        self.side = int(math.isqrt(args.n))
        self.cluster_spectrum: torch.Tensor | None = None
        self.dense_base: torch.Tensor | None = None
        self.edge_src: torch.Tensor | None = None
        self.edge_dst: torch.Tensor | None = None
        self.edge_weight: torch.Tensor | None = None
        if args.operator in {"diagonal-clusters", "hadamard-clusters"}:
            self.cluster_spectrum = make_cluster_spectrum(
                args.n,
                args.clusters,
                args.condition,
                args.cluster_width,
                device,
                dtype,
                args.seed + 17,
            )
        if args.operator == "planted-partition":
            self.dense_base = planted_partition_laplacian(
                args.n,
                args.communities,
                args.boundary_weight,
                device,
                dtype,
            )
        if args.operator == "sbm-laplacian":
            self.dense_base = sbm_laplacian(
                args.n,
                args.communities,
                args.p_in,
                args.p_out,
                args.seed + 19,
                device,
                dtype,
            )
        if args.operator == "sbm-sparse":
            self.edge_src, self.edge_dst, self.edge_weight = sparse_sbm_edges(
                args.n,
                args.communities,
                args.p_in,
                args.p_out,
                args.seed + 19,
                device,
                dtype,
            )
        if args.operator == "weak-tiles" and self.side * self.side != args.n:
            raise ValueError("--n must be a perfect square for weak-tiles")
        if args.operator == "hadamard-clusters" and args.n & (args.n - 1):
            raise ValueError("--n must be a power of two for hadamard-clusters")

    def base_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if self.args.operator == "diagonal-clusters":
            assert self.cluster_spectrum is not None
            return self.cluster_spectrum * vector
        if self.args.operator == "hadamard-clusters":
            assert self.cluster_spectrum is not None
            transformed = fwht(vector) / math.sqrt(self.n)
            return fwht(self.cluster_spectrum * transformed) / math.sqrt(self.n)
        if self.args.operator == "weak-tiles":
            return laplacian_tiles_base(vector, self.side, self.args.tile_size, self.args.boundary_weight)
        if self.args.operator in {"planted-partition", "sbm-laplacian"}:
            assert self.dense_base is not None
            return self.dense_base @ vector
        if self.args.operator == "sbm-sparse":
            assert self.edge_src is not None and self.edge_dst is not None and self.edge_weight is not None
            adjacency_times_vector = torch.zeros_like(vector)
            adjacency_times_vector.index_add_(0, self.edge_src, self.edge_weight * vector[self.edge_dst])
            return vector - adjacency_times_vector
        raise ValueError(f"unknown operator {self.args.operator!r}")

    def matvec(self, vector: torch.Tensor, log_mass: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_mass) * vector + torch.exp(log_scale) * self.base_matvec(vector)

    def spectral_upper(self) -> float:
        if self.args.operator in {"diagonal-clusters", "hadamard-clusters"}:
            assert self.cluster_spectrum is not None
            max_base = float(torch.max(self.cluster_spectrum.detach()).cpu())
            return self.args.mass + self.args.scale * max_base * (1.0 + 1.0e-6)
        if self.args.operator == "weak-tiles":
            return self.args.mass + 8.0 * self.args.scale
        if self.args.operator in {"planted-partition", "sbm-laplacian", "sbm-sparse"}:
            return self.args.mass + 2.0 * self.args.scale
        raise ValueError(f"unknown operator {self.args.operator!r}")



def tridiagonal(alphas: list[torch.Tensor], betas: list[torch.Tensor]) -> torch.Tensor:
    diagonal = torch.stack(alphas)
    tri = torch.diag(diagonal)
    if len(alphas) > 1:
        off = torch.stack(betas[: len(alphas) - 1])
        tri = tri + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    return tri



def pro_signed_roundoff(value: float, roundoff: float) -> float:
    if value > 0.0:
        return value + roundoff
    if value < 0.0:
        return value - roundoff
    return value



def update_pro_estimates_next(
    omega_prev: list[float],
    omega_curr: list[float],
    step: int,
    alpha: list[float],
    beta: list[float],
    eps1: float,
    anorm: float,
) -> list[float]:
    roundoff = eps1 * anorm
    inverse_beta = 1.0 / beta[step]
    omega_next = [0.0] * len(omega_curr)

    value = beta[0] * omega_curr[1] + (alpha[0] - alpha[step]) * omega_curr[0] - beta[step - 1] * omega_prev[0]
    omega_next[0] = inverse_beta * pro_signed_roundoff(value, roundoff)

    for k in range(1, step - 1):
        value = (
            beta[k] * omega_curr[k + 1]
            + (alpha[k] - alpha[step]) * omega_curr[k]
            + beta[k - 1] * omega_curr[k - 1]
            - beta[step - 1] * omega_prev[k]
        )
        omega_next[k] = inverse_beta * pro_signed_roundoff(value, roundoff)

    omega_next[step - 1] = 0.0
    omega_next[step] = eps1
    return omega_next



def pro_thresholds(args: argparse.Namespace, dtype: torch.dtype, n: int, steps: int) -> tuple[float, float, float, float, float, str]:
    eps_basis = torch.finfo(dtype).eps
    eps_matvec = float(args.pro_matvec_error) if args.pro_matvec_error > 0.0 else eps_basis
    if args.pro_profile == "historical":
        delta = math.sqrt(eps_basis / max(float(steps), 1.0))
        eta = eps_basis**0.75 / math.sqrt(max(float(steps), 1.0))
        eps1 = 0.5 * math.sqrt(float(n)) * eps_basis
    elif args.pro_profile == "unscaled":
        delta = math.sqrt(eps_basis)
        eta = eps_basis**0.75
        eps1 = eps_matvec
    elif args.pro_profile == "application":
        eps1 = eps_matvec
        delta = max(args.pro_delta_scale * math.sqrt(eps_basis), 8.0 * eps1)
        eta = max(args.pro_eta_scale * eps_basis**0.75, 2.0 * eps1)
        if not eta < delta:
            eta = 0.5 * delta
    else:
        raise ValueError(f"unknown PRO profile {args.pro_profile!r}")
    warnings: list[str] = []
    if not eps1 < eta:
        warnings.append("eps1>=eta")
    if not eta < delta:
        warnings.append("eta>=delta")
    return eps_basis, eps_matvec, eps1, delta, eta, ",".join(warnings)



def compute_pro_interval(omega: list[float], count: int, delta: float, eta: float) -> list[bool]:
    which = [False] * count
    found = False
    max_position = 0
    max_value = 0.0

    for i in range(count):
        value = abs(omega[i])
        if value >= delta:
            which[i] = True
            found = True
        if value > max_value:
            max_position = i
            max_value = value
    if not found and count > 0:
        which[max_position] = True

    for i in range(count):
        if not which[i]:
            continue
        for k in range(i - 1, -1, -1):
            if abs(omega[k]) < eta or which[k]:
                break
            which[k] = True
        for k in range(i + 1, count):
            if abs(omega[k]) < eta or which[k]:
                break
            which[k] = True
    return which



def reset_pro_estimates(omega: list[float], which: list[bool], eps1: float) -> None:
    for i, selected in enumerate(which):
        if selected:
            omega[i] = eps1



def orthogonalize_selected(work: torch.Tensor, basis: list[torch.Tensor], which: list[bool], passes: int = 1) -> torch.Tensor:
    selected = [vector for vector, keep in zip(basis, which) if keep]
    if not selected:
        return work
    q_block = torch.stack(selected, dim=1)
    for _ in range(passes):
        coeffs = q_block.T @ work
        work = work - q_block @ coeffs
    return work



def project_against_block(work: torch.Tensor, block: torch.Tensor, passes: int = 2) -> torch.Tensor:
    for _ in range(passes):
        coeffs = block.T @ work
        work = work - block @ coeffs
    return work



def project_against_tall_block(work: torch.Tensor, block: torch.Tensor, passes: int = 2) -> torch.Tensor:
    if block.numel() == 0:
        return work
    gram = block.T @ block
    ridge = 1.0e-6 * torch.clamp(torch.linalg.norm(gram), min=1.0)
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    for _ in range(passes):
        rhs = block.T @ work
        coeffs = torch.linalg.solve(gram, rhs)
        work = work - block @ coeffs
    return work



def lanczos(
    op: Operator,
    initial: torch.Tensor,
    log_mass: torch.Tensor,
    log_scale: torch.Tensor,
    steps: int,
    *,
    full_reorth: bool,
    store_basis: bool,
    check_orthogonality: bool,
    selective_reorth: bool = False,
    selective_tolerance: float = 1.0e-4,
    selective_check_interval: int = 8,
    selective_max_locked: int = 128,
    selective_duplicate_tolerance: float = 1.0e-5,
    partial_reorth: bool = False,
    partial_reorth_tolerance: float = 1.0e-3,
    pro_reorth: bool = False,
    pro_profile: str = "historical",
    pro_matvec_error: float = 0.0,
    pro_delta_scale: float = 1.0,
    pro_eta_scale: float = 1.0,
    periodic_exact: bool = False,
    periodic_reorth_period: int = 8,
    periodic_correlation_threshold: float = 1.0e-3,
    periodic_always_purge: bool = True,
    ritz_selective: bool = False,
    ritz_check_period: int = 8,
    ritz_residual_tolerance: float = 1.0e-3,
    ritz_minimum_quadrature_weight: float = 1.0e-10,
    ritz_maximum_locked_vectors: int = 128,
    full_reorth_passes: int = 1,
    beta_breakdown_tolerance: float = 0.0,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, int, LanczosStats]:
    q = initial / torch.linalg.norm(initial)
    q_previous = torch.zeros_like(q)
    beta_previous = torch.zeros((), dtype=initial.dtype, device=initial.device)
    basis: list[torch.Tensor] = []
    locked_vectors: list[torch.Tensor] = []
    locked_values: list[float] = []
    reorthogonalization_count = 0
    stats = LanczosStats(pro_rows=[] if pro_reorth else None)
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    beta_history: list[torch.Tensor] = []
    omega_prev = [0.0] * steps
    omega_curr = [0.0] * steps
    threshold_args = argparse.Namespace(
        pro_profile=pro_profile,
        pro_matvec_error=pro_matvec_error,
        pro_delta_scale=pro_delta_scale,
        pro_eta_scale=pro_eta_scale,
    )
    pro_eps_basis, pro_eps_matvec, pro_eps1, pro_delta, pro_eta, pro_warning = pro_thresholds(
        threshold_args, initial.dtype, initial.numel(), steps
    )
    pro_anorm = 1.0
    pro_force_reorth = False
    pro_force_full_reorth = False
    pro_force_columns = [False] * steps
    for step in range(steps):
        if (
            full_reorth
            or selective_reorth
            or partial_reorth
            or pro_reorth
            or periodic_exact
            or ritz_selective
            or store_basis
            or check_orthogonality
        ):
            basis.append(q)
        matvec_start = time.perf_counter()
        work = op.matvec(q, log_mass, log_scale)
        stats.matvec_seconds += time.perf_counter() - matvec_start
        alpha = torch.dot(q, work)
        work = work - alpha * q
        if step > 0:
            work = work - beta_previous * q_previous
        if selective_reorth and locked_vectors:
            locked_block = torch.stack(locked_vectors, dim=1)
            coeffs = locked_block.T @ work
            work = work - locked_block @ coeffs
        if full_reorth or pro_force_full_reorth:
            orth_start = time.perf_counter()
            q_block = torch.stack(basis, dim=1)
            work = project_against_block(work, q_block, passes=full_reorth_passes)
            stats.orthogonalization_seconds += time.perf_counter() - orth_start
            reorthogonalization_count += 1
            stats.record_event(step + 1, step + 1)
        elif partial_reorth and basis:
            q_block = torch.stack(basis, dim=1)
            coeffs = q_block.T @ work
            if torch.linalg.norm(coeffs) > partial_reorth_tolerance * torch.clamp(torch.linalg.norm(work), min=1.0e-30):
                orth_start = time.perf_counter()
                work = work - q_block @ coeffs
                reorthogonalization_count += 1
                coeffs = q_block.T @ work
                work = work - q_block @ coeffs
                stats.orthogonalization_seconds += time.perf_counter() - orth_start
                stats.record_event(step + 1, step + 1)
        beta = torch.linalg.norm(work)
        alphas.append(alpha)
        alpha_value = float(alpha.detach().cpu())
        beta_value = float(beta.detach().cpu()) if bool(torch.isfinite(beta).detach().cpu()) else float("nan")
        alpha_values.append(alpha_value)
        if periodic_exact and (step + 1) % max(1, periodic_reorth_period) == 0 and basis:
            q_block = torch.stack(basis, dim=1)
            beta_for_corr = torch.clamp(beta, min=torch.tensor(1.0e-30, dtype=beta.dtype, device=beta.device))
            coeffs = q_block.T @ work
            max_corr = float((torch.max(torch.abs(coeffs)) / beta_for_corr).detach().cpu()) if coeffs.numel() else 0.0
            if periodic_always_purge or max_corr > periodic_correlation_threshold:
                orth_start = time.perf_counter()
                work = work - q_block @ coeffs
                coeffs2 = q_block.T @ work
                work = work - q_block @ coeffs2
                stats.orthogonalization_seconds += time.perf_counter() - orth_start
                reorthogonalization_count += 1
                stats.record_event(step + 1, step + 1)
                beta = torch.linalg.norm(work)
                beta_value = float(beta.detach().cpu()) if bool(torch.isfinite(beta).detach().cpu()) else float("nan")

        if (
            ritz_selective
            and step + 1 < steps
            and step + 1 >= 4
            and (step + 1) % max(1, ritz_check_period) == 0
            and basis
        ):
            with torch.no_grad():
                tri_probe = tridiagonal(alphas, betas).detach().cpu()
                theta, z = torch.linalg.eigh(tri_probe)
                beta_cpu = beta.detach().cpu()
                rho = torch.abs(beta_cpu * z[-1, :])
                weights = z[0, :] * z[0, :]
                anorm_probe = max(pro_anorm, float(torch.linalg.norm(tri_probe, ord=2)))
                thresholds = ritz_residual_tolerance * torch.maximum(
                    torch.full_like(theta, anorm_probe), torch.abs(theta)
                )
                selected = (rho <= thresholds) & (weights >= ritz_minimum_quadrature_weight)
                selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
                if selected_indices.numel() > ritz_maximum_locked_vectors:
                    order = torch.argsort(rho[selected_indices])
                    selected_indices = selected_indices[order[:ritz_maximum_locked_vectors]]
                if selected_indices.numel() > 0:
                    orth_start = time.perf_counter()
                    q_matrix = torch.stack([item.detach() for item in basis], dim=1)
                    y = q_matrix @ z[:, selected_indices].to(device=q_matrix.device, dtype=q_matrix.dtype)
                    work = project_against_tall_block(work, y, passes=2)
                    stats.orthogonalization_seconds += time.perf_counter() - orth_start
                    reorthogonalization_count += 1
                    stats.record_event(int(selected_indices.numel()), step + 1)
                    beta = torch.linalg.norm(work)
                    beta_value = float(beta.detach().cpu()) if bool(torch.isfinite(beta).detach().cpu()) else float("nan")

        if pro_reorth and not pro_force_full_reorth:
            if step > 0:
                pro_anorm = max(pro_anorm, abs(alpha_value) + beta_value + beta_values[step - 1])
            else:
                pro_anorm = max(pro_anorm, abs(alpha_value) + beta_value)
            beta_values.append(beta_value)

            should_reorthogonalize = False
            max_omega_est = 0.0
            max_omega_actual = float("nan")
            floor_ratio = float("inf")
            if step > 0 and math.isfinite(beta_value) and beta_value > 0.0:
                omega_next = update_pro_estimates_next(
                    omega_prev, omega_curr, step, alpha_values, beta_values, pro_eps1, pro_anorm
                )
                omega_prev, omega_curr = omega_curr, omega_next
                max_omega_est = max((abs(omega_curr[i]) for i in range(step)), default=0.0)
                should_reorthogonalize = any(abs(omega_curr[i]) > pro_delta for i in range(step))
                floor_ratio = pro_anorm * pro_eps1 / max(beta_value * pro_delta, 1.0e-300)
                if basis:
                    q_block = torch.stack([item.detach() for item in basis], dim=1)
                    actual = (q_block.T @ work.detach()) / max(beta_value, 1.0e-30)
                    max_omega_actual = float(torch.max(torch.abs(actual)).detach().cpu()) if actual.numel() else 0.0

            selected_count_step = 0
            selected_fraction_step = 0.0
            trigger_from_force_next = pro_force_reorth
            if should_reorthogonalize or pro_force_reorth:
                if pro_force_reorth:
                    pro_force_reorth = False
                    which = pro_force_columns[: step + 1]
                    for i in range(step + 1):
                        pro_force_columns[i] = False
                    which[step] = True
                else:
                    pro_force_reorth = True
                    which = compute_pro_interval(omega_curr, step, pro_delta, pro_eta) + [True]
                    for i in range(step):
                        pro_force_columns[i] = which[i]
                    pro_force_columns[step] = True
                    if step + 1 < steps:
                        pro_force_columns[step + 1] = True
                reset_pro_estimates(omega_curr, which, pro_eps1)
                orth_start = time.perf_counter()
                work = orthogonalize_selected(work, basis, which, passes=2)
                stats.orthogonalization_seconds += time.perf_counter() - orth_start
                reorthogonalization_count += 1
                selected_count_step = sum(1 for item in which if item)
                selected_fraction_step = selected_count_step / float(step + 1)
                stats.record_event(selected_count_step, step + 1)
                beta = torch.linalg.norm(work)
                beta_value = float(beta.detach().cpu()) if bool(torch.isfinite(beta).detach().cpu()) else float("nan")
                beta_values[step] = beta_value

            trigger_from_floor = math.isfinite(beta_value) and beta_value * pro_delta < pro_anorm * pro_eps1
            if stats.pro_rows is not None:
                stats.pro_rows.append(
                    {
                        "step": step,
                        "eps_basis": pro_eps_basis,
                        "eps_matvec": pro_eps_matvec,
                        "eps1": pro_eps1,
                        "delta": pro_delta,
                        "eta": pro_eta,
                        "eps1_over_delta": pro_eps1 / pro_delta,
                        "floor_ratio": floor_ratio,
                        "max_omega_est": max_omega_est,
                        "max_omega_actual": max_omega_actual,
                        "trigger_from_omega": should_reorthogonalize,
                        "trigger_from_force_next": trigger_from_force_next,
                        "trigger_from_floor_fallback": trigger_from_floor,
                        "selected_count": selected_count_step,
                        "selected_fraction": selected_fraction_step,
                        "warning": pro_warning,
                    }
                )
            if trigger_from_floor:
                pro_force_full_reorth = True
        if (
            selective_reorth
            and step + 1 < steps
            and step + 1 >= 4
            and (step + 1) % max(1, selective_check_interval) == 0
            and len(locked_vectors) < selective_max_locked
        ):
            with torch.no_grad():
                tri_probe = tridiagonal(alphas, betas).detach().cpu()
                eigenvalues, eigenvectors = torch.linalg.eigh(tri_probe)
                residuals = torch.abs(beta.detach().cpu() * eigenvectors[-1, :])
                threshold = selective_tolerance * max(float(torch.linalg.norm(tri_probe, ord=2)), 1.0)
                q_matrix = torch.stack([item.detach() for item in basis], dim=1)
                for residual, value, vector_coeffs in sorted(
                    zip(residuals.tolist(), eigenvalues.tolist(), eigenvectors.T),
                    key=lambda item: item[0],
                ):
                    if len(locked_vectors) >= selective_max_locked:
                        break
                    if residual > threshold:
                        break
                    if any(abs(value - old) <= selective_duplicate_tolerance * max(abs(value), abs(old), 1.0) for old in locked_values):
                        continue
                    locked = q_matrix @ vector_coeffs.to(device=q_matrix.device, dtype=q_matrix.dtype)
                    if locked_vectors:
                        locked_block = torch.stack(locked_vectors, dim=1)
                        locked = locked - locked_block @ (locked_block.T @ locked)
                    locked_norm = torch.linalg.norm(locked)
                    if locked_norm <= 1.0e-8:
                        continue
                    locked = locked / locked_norm
                    locked_vectors.append(locked.detach())
                    locked_values.append(float(value))
        beta_history.append(beta.detach())
        if step + 1 == steps:
            break
        if (not bool(torch.isfinite(beta).detach().cpu())) or float(beta.detach().cpu()) <= beta_breakdown_tolerance:
            break
        betas.append(beta)
        if not pro_reorth:
            beta_values.append(float(beta.detach().cpu()))
        q_previous = q
        q = work / beta
        beta_previous = beta

    tri = tridiagonal(alphas, betas)
    if check_orthogonality and basis:
        q_matrix = torch.stack(basis, dim=1)
        gram = q_matrix.T @ q_matrix
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        off = gram - eye
        orth = torch.linalg.norm(off)
        stats.max_gram_offdiag = float(torch.max(torch.abs(off)).detach().cpu())
    else:
        orth = torch.full((), float("nan"), dtype=initial.dtype, device=initial.device)
    stats.steps_done = len(alphas)
    return tri, basis, orth, torch.stack(beta_history), len(locked_vectors) + reorthogonalization_count, stats



def chebyshev_adjoint_affine(
    op: Operator,
    initial: torch.Tensor,
    log_mass: torch.Tensor,
    log_scale: torch.Tensor,
    args: argparse.Namespace,
    coeffs: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[float, torch.Tensor, float, float]:
    coeffs = coeffs.to(device=initial.device, dtype=initial.dtype)
    degree = coeffs.numel() - 1
    mass = torch.exp(log_mass)
    scale_param = torch.exp(log_scale)
    interval_scale = 2.0 / (upper - lower)

    def scaled_apply(vector: torch.Tensor) -> torch.Tensor:
        return (2.0 * op.matvec(vector, log_mass, log_scale) - (upper + lower) * vector) / (upper - lower)

    with torch.no_grad():
        states: list[torch.Tensor] = [initial]
        value = coeffs[0] * torch.dot(initial, initial)
        if degree >= 1:
            states.append(scaled_apply(initial))
            value = value + coeffs[1] * torch.dot(initial, states[1])
            for order in range(1, degree):
                next_state = 2.0 * scaled_apply(states[order]) - states[order - 1]
                states.append(next_state)
                value = value + coeffs[order + 1] * torch.dot(initial, next_state)

        gradient = torch.zeros(2, dtype=initial.dtype, device=initial.device)
        if degree >= 1:
            lambda_next2 = torch.zeros_like(initial)
            lambda_next1 = coeffs[degree] * initial
            for order in range(degree - 1, 0, -1):
                state = states[order]
                gradient[0] = gradient[0] + 2.0 * interval_scale * mass * torch.dot(lambda_next1, state)
                gradient[1] = gradient[1] + 2.0 * interval_scale * scale_param * torch.dot(
                    lambda_next1, op.base_matvec(state)
                )
                current = coeffs[order] * initial + 2.0 * scaled_apply(lambda_next1) - lambda_next2
                lambda_next2 = lambda_next1
                lambda_next1 = current
            state0 = states[0]
            gradient[0] = gradient[0] + interval_scale * mass * torch.dot(lambda_next1, state0)
            gradient[1] = gradient[1] + interval_scale * scale_param * torch.dot(lambda_next1, op.base_matvec(state0))
    return float(value.detach().cpu()), gradient, float("nan"), float("nan")
