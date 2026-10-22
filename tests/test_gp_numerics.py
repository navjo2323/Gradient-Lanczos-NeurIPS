"""Independent dense checks for the GP projected gradient and preconditioner."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
pytest.importorskip("jax")
from experiments.gp_training import train as gp


def test_small_log_frechet_matches_dense_autograd():
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(2, 6, 6, generator=generator, dtype=torch.float64)
    t = (a @ a.transpose(-1, -2) + torch.eye(6)).requires_grad_()
    value, actual_g, _ = gp._projected_logdet_and_g(t, n=17)
    evals, evecs = torch.linalg.eigh(t)
    oracle = (17 * (evecs[:, 0, :].square() * torch.log(evals)).sum(-1)).mean()
    expected_g, = torch.autograd.grad(oracle, t)
    torch.testing.assert_close(value, oracle, atol=1e-12, rtol=1e-12)
    # The experiment averages across probes during the large-matrix contraction.
    torch.testing.assert_close(actual_g / 2, expected_g, atol=3e-12, rtol=3e-12)


def test_preconditioner_inverse_sqrt_and_derivative():
    generator = torch.Generator().manual_seed(7)
    root = torch.randn(7, 3, generator=generator, dtype=torch.float64).requires_grad_()
    diagonal = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    rhs = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    actual = gp._low_rank_plus_diag_inv_sqrt_apply(root, diagonal, rhs)
    p = root @ root.T + diagonal * torch.eye(7, dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(p.detach())
    expected = (vecs * vals.rsqrt()) @ vecs.T @ rhs
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert torch.autograd.gradcheck(lambda r, d: gp._low_rank_plus_diag_inv_sqrt_apply(r, d, rhs),
                                   (root, diagonal), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_lanczos_recurrence_matches_dense_action():
    generator = torch.Generator().manual_seed(6)
    a = torch.randn(12, 12, generator=generator, dtype=torch.float64)
    a = a + a.T + 8 * torch.eye(12)
    probes = torch.randn(3, 12, generator=generator, dtype=torch.float64)
    q, t = gp._lanczos_qt_no_reorth_matmul(lambda v: a @ v, probes, 6)
    # The residual is confined to the final column of each independent recurrence.
    for qi, ti in zip(q, t):
        residual = a @ qi - qi @ ti
        torch.testing.assert_close(residual[:, :-1], torch.zeros_like(residual[:, :-1]),
                                   atol=2e-13, rtol=0)


def test_stored_paper_table_and_training_endpoints():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "results/paper/gp_training"
    data = json.loads((root / "training.json").read_text())
    by_method = {m["method"]: m for m in data["methods"]}
    np.testing.assert_allclose(by_method["gpytorch"]["final_rmse"], 0.7446171847951735, rtol=0, atol=1e-14)
    np.testing.assert_allclose(by_method["ours_no_reorth"]["final_rmse"], 0.6887978068780842, rtol=0, atol=1e-14)
    assert all(len(m["history"]) == 120 for m in data["methods"])
