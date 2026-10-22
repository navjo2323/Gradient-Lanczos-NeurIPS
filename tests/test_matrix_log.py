"""Numerical checks for the Figure 1 driver, including its independent reference."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("matfree")
from experiments.matrix_log import run as ml

jax.config.update("jax_enable_x64", True)


def test_blocked_kernel_matches_dense_with_padding():
    x, _ = ml.make_toy_regression(jax.random.PRNGKey(3), 9, 2)
    params = ml.init_params()
    rhs = jax.random.normal(jax.random.PRNGKey(4), (9, 3), dtype=ml.jnp.float64)
    actual = ml.kernel_matmat_blocked(params, x, rhs, jitter=1e-6, row_block_size=4)
    expected = ml.kernel_matrix(params, x, 1e-6) @ rhs
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_projected_log_derivative_against_independent_logm_difference():
    from scipy.linalg import logm
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6))
    t = a @ a.T + np.eye(6)
    e = rng.normal(size=(6, 6))
    e = (e + e.T) / 2
    g = np.asarray(ml.divided_difference_log_matrix(ml.jnp.asarray(t), 2.3))
    h = 1e-5
    oracle = 2.3 * (logm(t + h * e)[0, 0] - logm(t - h * e)[0, 0]) / (2 * h)
    np.testing.assert_allclose(np.sum(g * e), oracle, rtol=3e-7, atol=1e-8)


def test_shared_forward_and_converged_gradient():
    config = ml.Config(n=32, d=2, depths=[24], num_probes=1, seed=0,
                       jitter=1e-6, reference="dense", reference_depth=32,
                       dense_threshold=32, outdir="unused", row_block_size=16,
                       timing_repeats=1, stable_log_divdiff=True, clip_ritz_to_gp_floor=False)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x, _ = ml.make_toy_regression(k1, 32, 2)
    probes = ml.make_rademacher_probes(k2, 1, 32)
    params = ml.init_params()
    value, vs, ws = ml.repo_forward_value_and_frozen_VW(params, x, probes, config, 24, reortho="full")
    adj_value, _ = ml.repo_adjoint_value_and_grad(params, x, probes, config, 24, reortho="full")
    _, reference_grad = ml.dense_value_and_grad(params, x, probes, config)
    actual_grad = ml.ours_grad_from_frozen_VW(params, x, vs, ws, config)
    assert value == adj_value
    assert ml.safe_rel_grad_error(actual_grad, reference_grad) < 1e-7
