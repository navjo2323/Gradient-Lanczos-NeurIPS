# Projected gradient API

`src/lanczos_gradients/core.py` is a compact NumPy reference implementation of
Algorithm 1 and the blockwise replay procedure in Section 4.3. It was added for
this release. The historical experiment implementations are preserved separately
in `experiments/`; paper results were not generated with this new API.

For a real symmetric operator `A(theta)`, pass:

- `matvec(v)`: `A(theta) @ v` at a fixed parameter point;
- `vjp(v, w)`: the parameter vector with entries `w.T @ dA/dtheta_j @ v`,
  holding `v` and `w` fixed;
- the starting vector, depth, scalar function and its derivative.

```python
value, gradient = value_and_grad(
    matvec, vjp, u, depth=32, f=np.exp, df=np.exp,
    mode="replay", block_size=8,
)
```

The gradient contracts `V G V.T` with the operator derivatives. It approximates
the derivative of the matrix-function quadratic form and generally differs from
the exact derivative of finite Lanczos quadrature. The Lanczos basis and `G` are
held fixed when calling the VJP. Explicit parameter dependence in the starting
vector or in `f` requires additional derivative terms, which this API does not add.

Stored mode keeps `V` and forms `W=V@G`. Optional full reorthogonalization uses
two passes. Replay mode first saves only the start and tridiagonal coefficients.
For each column block it reconstructs the unreorthogonalized recurrence and
accumulates the corresponding columns of `V@G`. Use a deterministic matvec at
the same parameter point for all passes. Reorthogonalization is incompatible
with this replay implementation.

Vector storage is O(nm) in stored mode and O(nb) in replay mode, plus O(m²)
for small matrices and the callback's own workspace. Replay performs up to
ceil(m/b) additional passes; it is a storage/computation tradeoff. No runtime
comparison for general replay is asserted by the archived experiments.

The API accepts real float64 arrays. The Hamiltonian experiment uses its own
complex polarization into real cosine/sine problems. GPU experiments use the
specialized affine formula, which avoids a stored or replayed basis.

`python examples/quickstart.py` compares both modes with a dense SciPy Fréchet
oracle. Tests also check full-depth exponential and logarithm derivatives,
invariant-subspace breakdown, repeated eigenvalues and several replay block sizes.
