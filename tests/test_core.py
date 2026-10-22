import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet, logm
from lanczos_gradients import lanczos, projected_sensitivity, value_and_grad


def problem(n=16):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(n, n))
    a = (a+a.T)/(3*np.sqrt(n))
    e = rng.normal(size=(3, n, n))
    e = (e+e.transpose(0, 2, 1))/(3*np.sqrt(n))
    u = rng.normal(size=n)
    vjp = lambda v, w: np.einsum('i,kij,j->k', w, e, v)
    return a, e, u, vjp


def test_full_depth_dense_frechet():
    a, e, u, vjp = problem()
    value, gradient = value_and_grad(lambda v: a@v, vjp, u, len(u), np.exp, np.exp,
                                     reorthogonalize=True)
    exact = [u@expm_frechet(a, ej, compute_expm=False)@u for ej in e]
    np.testing.assert_allclose(value, u@expm(a)@u, rtol=2e-13)
    np.testing.assert_allclose(gradient, exact, rtol=3e-12, atol=1e-12)


@pytest.mark.parametrize('block_size', [1, 3, 8, 30])
def test_replay_matches_stored(block_size):
    a, e, u, vjp = problem()
    stored = value_and_grad(lambda v: a@v, vjp, u, 8, np.exp, np.exp)
    replay = value_and_grad(lambda v: a@v, vjp, u, 8, np.exp, np.exp,
                            mode='replay', block_size=block_size)
    np.testing.assert_allclose(replay[0], stored[0], rtol=1e-14)
    np.testing.assert_allclose(replay[1], stored[1], rtol=2e-13, atol=1e-13)


def test_log_gradient_against_dense_central_difference():
    a, e, u, vjp = problem(10)
    a += 3*np.eye(len(u))
    value, gradient = value_and_grad(lambda v: a@v, vjp, u, len(u), np.log,
                                     lambda x: 1/x, reorthogonalize=True)
    eps = 1e-5
    exact = [u@(logm(a+eps*ej)-logm(a-eps*ej))@u/(2*eps) for ej in e]
    np.testing.assert_allclose(value, u@logm(a)@u, rtol=1e-12)
    np.testing.assert_allclose(gradient, exact, rtol=1e-7, atol=2e-8)


def test_invariant_subspace_breakdown():
    u = np.arange(1., 10.)
    r = lanczos(lambda v: 2*v, u, 9, store_basis=False)
    assert r.basis is None and len(r.alpha) == 1 and len(r.beta) == 0
    for mode in ['stored', 'replay']:
        value, gradient = value_and_grad(lambda v: 2*v, lambda v,w: np.array([w@v]),
                                         u, 9, np.exp, np.exp, mode=mode)
        np.testing.assert_allclose([value, gradient[0]], np.exp(2)*(u@u))


def test_repeated_eigenvalue_divided_difference():
    from lanczos_gradients import LanczosResult
    r = LanczosResult(np.array([2., 2.]), np.array([0.]), 3., np.array([1., 0.]), None)
    value, g = projected_sensitivity(r, np.exp, np.exp)
    np.testing.assert_allclose(value, 3*np.exp(2))
    np.testing.assert_allclose(g, np.diag([3*np.exp(2), 0]), atol=1e-14)


def test_invalid_replay_reorthogonalization():
    with pytest.raises(ValueError, match='unreorthogonalized'):
        value_and_grad(lambda v: v, lambda v,w: w@v, np.ones(3), 2, np.exp,
                       np.exp, mode='replay', reorthogonalize=True)
