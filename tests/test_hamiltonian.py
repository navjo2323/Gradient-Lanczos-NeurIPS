"""Independent dense checks for the released Hamiltonian gradient."""
import json
from pathlib import Path
import numpy as np
from experiments.hamiltonian.train import make_dataset, dense_loss_and_grad, gm_loss_and_grad
from experiments.hamiltonian.plot import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_dense_gradient_finite_difference_and_polarized_full_depth():
    basis, _, data = make_dataset(3, 2, .2, .7, 3, .05, 'easy')
    theta = data.theta_true + np.random.default_rng(999).normal(size=8) * .1
    loss, gradient = dense_loss_and_grad(theta, basis, data)
    direction = np.random.default_rng(4).normal(size=8)
    direction /= np.linalg.norm(direction)
    step = 1e-6
    plus = dense_loss_and_grad(theta + step * direction, basis, data)[0]
    minus = dense_loss_and_grad(theta - step * direction, basis, data)[0]
    np.testing.assert_allclose(gradient @ direction, (plus - minus)/(2*step), rtol=1e-6, atol=1e-9)
    projected_loss, projected_gradient = gm_loss_and_grad(theta, basis, data, m=8)
    np.testing.assert_allclose(projected_loss, loss, atol=1e-14)
    np.testing.assert_allclose(projected_gradient, gradient, rtol=1e-10, atol=1e-11)


def test_saved_table_2_matches_paper_and_parameter_vectors():
    data = json.loads((ROOT / 'results/paper/hamiltonian/results.json').read_text())
    rows = summarize(data)
    assert [f"{r['speedup']:.1f}" for r in rows] == ['1.0', '14.8', '12.4', '10.4']
    assert [f"{r['final_parameter_error']:.2e}" for r in rows] == ['2.00e-10','5.43e-03','1.61e-04','1.05e-07']
    assert [f"{r['initial_gradient_relative_error']:.2e}" for r in rows[1:]] == ['7.49e-02','6.31e-04','5.62e-07']
    truth = np.array(data['theta_true'])
    np.testing.assert_allclose(np.linalg.norm(np.array(data['theta_dense'])-truth), rows[0]['final_parameter_error'], rtol=1e-12)
    for row, run in zip(rows[1:],data['gm_runs'].values()):
        np.testing.assert_allclose(np.linalg.norm(np.array(run['theta'])-truth), row['final_parameter_error'], rtol=1e-12)
        assert len(run['history']) == 400
