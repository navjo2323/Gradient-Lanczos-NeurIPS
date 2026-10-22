"""A small dense oracle for the matrix-free stored/replay API."""
import numpy as np
from scipy.linalg import expm, expm_frechet
from lanczos_gradients import value_and_grad


def main():
    rng = np.random.default_rng(17)
    n = 24
    a = rng.normal(size=(n, n))
    a = (a + a.T) / (4 * np.sqrt(n))
    e = rng.normal(size=(3, n, n))
    e = (e + e.transpose(0, 2, 1)) / (4 * np.sqrt(n))
    u = rng.normal(size=n)
    vjp = lambda v, w: np.einsum("i,kij,j->k", w, e, v)
    reference = np.array([u @ expm_frechet(a, ej, compute_expm=False) @ u for ej in e])
    values = []
    for mode in ("stored", "replay"):
        value, gradient = value_and_grad(lambda v: a @ v, vjp, u, 12,
                                         np.exp, np.exp, mode=mode, block_size=4)
        error = np.linalg.norm(gradient-reference) / np.linalg.norm(reference)
        print(f"{mode}: value={value:.12g}, relative gradient error={error:.3e}")
        np.testing.assert_allclose(gradient, reference, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(value, u @ expm(a) @ u, rtol=1e-11)
        values.append((value, gradient))
    np.testing.assert_allclose(values[0][1], values[1][1], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    main()
