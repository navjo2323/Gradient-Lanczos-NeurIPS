# Matrix-log quadratic form: Figure 1

`experiments/matrix_log/run.py` computes the projected gradient and the Matfree
Lanczos-adjoint baseline using the same Lanczos forward computation for each
reorthogonalization setting. The dense reference uses an eigendecomposition and
the matrix-log Fréchet divided difference; it does not differentiate eigenvectors.
The driver was copied from the original experiment and its numerical functions
are unchanged.

Install the repository's experiment dependencies, then run from the repository
root. A small example requires no downloaded data:

```sh
python scripts/run_experiment.py configs/matrix_log_quick.json
```

The complete Figure 1 experiment is:

```sh
python scripts/run_experiment.py configs/matrix_log_paper.json
```

The full configuration uses `n=15000`, two features, one fixed Rademacher probe,
seed 0, row blocks of 512, and depths
`10,20,30,40,50,60,80,100,150,200,250,300`. Both full reorthogonalization and no
reorthogonalization are run in float64. The covariance is an RBF kernel with
lengthscale 0.9, signal standard deviation 1.1, noise standard deviation 0.15,
and diagonal jitter `1e-6`. The objective is one fixed-probe quadratic form,
not a dense trace. The stable log divided difference is enabled; the optional
GP-floor Ritz clipping flag is disabled. The inherited implementation still
floors Ritz values at the smallest positive representable value before taking
a logarithm.

The full run explicitly requests a dense `15000 x 15000` reference. One float64
matrix alone occupies 1.8 GB; the eigensolver and reference calculations need
several such arrays. Use a machine with ample memory. The small example exercises
the same code paths and a dense reference, but does not reproduce the paper's
large-problem numbers.

Regenerate the figure directly from the archived results without rerunning the
numerical experiment:

```sh
python experiments/matrix_log/plot.py results/paper/matrix_log/results.json --out outputs/figures/figure1.png
python experiments/matrix_log/plot.py results/paper/matrix_log/results.json --out outputs/figures/figure1.pdf
```

`results/paper/matrix_log/results.json` contains every saved depth, gradients,
reference values, error measures, and original timings. `figure1_original.png`
is the original figure asset. Some no-reorthogonalization adjoint results became
nonfinite. Those entries remain missing measurements: the portable JSON encodes
them as `null`, and the plot leaves gaps. The exact original bytes, including
nonstandard JSON `NaN` tokens, are retained in `results.original.json.raw`.
No points have been interpolated or replaced with projected-method values.

The gradient curves are relative errors against the dense reference, not relative
errors between the two approximate gradients. At `m=60`, the projected and
adjoint full-reorthogonalization errors are about `5.19e-5` and `1.96e-5`; at
`m=80`, they are about `2.02e-8` and `3.89e-9`. Original dependency versions were
not recorded in the result file. The release environment and validation results
are documented separately from this historical record. In particular, historical
nan locations and timings can depend on numerical-library versions and hardware.

The upstream Matfree package implements the baseline. Attribution and the
original experiment collection's MIT notice are retained under
`third_party/matfree_extensions/`; no local extension modules are required.
Exact source-file hashes and changes are in
`docs/gp_matrix_log_source_manifest.json`.

For a check at the actual paper size without allocating the dense eigensolver
reference again, run:

```sh
python experiments/matrix_log/validate_saved_reference.py --check
```

This reruns depths 60 and 80 for both methods and both reorthogonalization modes
with the archived `n=15000` inputs/seeds/settings. It compares fresh values and
gradients with the archived runs and uses the saved dense-reference gradient to
normalize errors. It does **not** recompute an independent dense reference. All
fresh coordinates and discrepancies are written to
`outputs/matrix_log/paper_size_validation.json`. The archived timings total about
three minutes for these selected computations and about 28 minutes for the full
sweep, excluding the unrecorded dense-reference cost; actual runtime depends on
the environment.

Release validation found that dependency versions affect the unreorthogonalized
recurrence at this size. With JAX/jaxlib 0.10.0 and Matfree 0.5.5, the selected
unreorthogonalized rerun reproduces the archived coordinates. A comparison with
JAX/jaxlib 0.10.2 and Matfree 0.6.3 changed the finite-depth no-reorthogonalization
results, including an adjoint becoming nonfinite at `m=80`. Full-reorthogonalization
results remained close to the archive. Use the pinned release environment for
reproduction. These observed sensitivities also mean that identical results
across different hardware/compiler stacks are not guaranteed. The original
result files did not record dependency versions; release pins are based on
observed reproduction checks, not inferred historical metadata.

The `--check` flag exits nonzero if a selected forward value differs by more
than `1e-10` relative, or a gradient differs by more than `1e-10` after division
by the archived dense gradient norm. This check is intended for the default
finite archived depth-60/80 cases. Omit the flag to record comparisons on another
numerical stack without enforcing that tolerance.

The final clean pinned-environment validation reproduced the archived forward
values and all projected/adjoint gradient coordinates exactly for `n=15000`,
`m=60,80`, with both full reorthogonalization and no reorthogonalization. The full
12-depth sweep and its large dense-reference eigendecomposition were not rerun
during release validation. The archive supplies those measurements; the complete
rerun command above remains available. See
`results/validation/gp_matrix_log.json` for the selected-size checks and the
separate newer-dependency comparison.
