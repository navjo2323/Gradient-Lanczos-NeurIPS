# Affine covariance model: Figures 7–8 and Tables 5–8

Run these commands from the repository root. The plot command needs only the base
dependencies; computation and verification also need `pip install -e '.[gpu]'`.

```sh
python -m experiments.affine_gpu.plot
python -m experiments.affine_gpu.verify_records
python -m experiments.affine_gpu.run --config configs/affine_gpu_quick.json
python -m experiments.affine_gpu.run --config configs/affine_gpu_quick.json --device mps --output results/reruns/affine_gpu/quick_mps.csv
```

The first command rebuilds the paper figures from the four archived CSVs in
`results/paper/affine_gpu/` and writes PDF/PNG figures under
`artifacts/affine_gpu/figures/`. It also writes CSV and TeX versions of Tables 5–8.
Table 8 uses the archived component-gradient diagnostics, which the verification
command recomputes independently. The saved CSVs have 45, 5, 4 and 36 rows at
depths 128, 384, 768 and 1536, respectively.

The verification command regenerates the seeded operator and samples on CPU,
checks spectral reference values and gradients for every saved parameter vector,
recomputes all reported error measures, and replays every saved Adam update.
It also checks the current full-reorthogonalization autograd graph on a tiny CPU
example. That inspection is evidence about the installed PyTorch version; its
allocation pattern is reported separately from the historical memory discussion.
Reports and rebuilt supplemental CSVs go to `artifacts/affine_gpu/verification/`.
This does not rerun historical GPU timings.

The quick configuration uses dimension 64, depth 8 and one Adam update for all
five methods. CPU is its default; use `--device mps` on Apple Silicon or
`--device cuda` on a compatible NVIDIA system. Large vectors use float32 and
small spectral calculations use CPU float64. The wrapper writes the actual
configuration, library versions, device, thread setting and relevant environment
variables to a `.metadata.json` alongside each new CSV.

The following explicit configurations perform new runs at paper problem sizes:

```sh
python -m experiments.affine_gpu.run --config configs/affine_gpu_m128_reconstructed.json
python -m experiments.affine_gpu.run --config configs/affine_gpu_m384_reconstructed.json
python -m experiments.affine_gpu.run --config configs/affine_gpu_m768_reconstructed.json
python -m experiments.affine_gpu.run --config configs/affine_gpu_m1536_reconstructed.json
```

They select MPS and require a GPU by default. Use `--device cpu` or `--device cuda`
to select another backend. The 128 and 1536 configurations perform eight updates
(nine evaluations); the 384 and 768 configurations evaluate initialization only.
Full-reorthogonalized AD is included at 128 and 384. It is omitted at 768 and
1536 to match the set of completed archived results. No OOM measurement is
fabricated for those missing rows.

These are **reconstructed configurations**, not recovered original launch
records. The original CSVs do not establish the PyTorch/MPS environment, Ritz
clipping choice, full-reorthogonalization pass count, or the Chebyshev degree
for depth 1536. The release explicitly selects the current source defaults:
`ritz_guard=none`, `full_reorth_passes=1`, and degree 4096 at depth 1536.
Degrees 128, 384 and 768 for the first three runs are established by the original
filenames. The beta breakdown threshold is explicitly 1e-30. Recomputed GPU
trajectories and timings may differ; the published figures are reproduced from
the archived measurements. The original OOM failure log is unavailable.

The operator is `A = mu I + s H diag(lambda) H.T`, with the normalized
Walsh–Hadamard transform `H`. Dimension is 32768, with 128 logarithmically
spaced cluster centers from 1e-10 to 1. Float32 spectrum generation uses a CPU
PyTorch generator with seed 17, multiplicative Gaussian jitter of scale 1e-8,
and a permutation from the same generator. The driver-level operator seed is 0;
the shared operator adds 17 internally. Seed 123 generates two samples in the
known spectral basis. True parameters are `(mu,s,alpha)=(1e-6,1,0.5)` and the
shared initialization is `(1e-5,5,1.5)`. The regularized negative log likelihood
uses an exact log determinant and a quadratic prior of strength 1e-4 centered
at the true coordinates `(log(mu),log(s),alpha)`. Adam uses learning rate 0.02.

The projected affine rule contracts the small sensitivity with `mu I` and
`T-mu I`, and computes the alpha derivative spectrally. It does not store or
replay a basis. This affine specialization therefore does not benchmark the
general stored-basis or replay contraction. The numerical bodies are preserved
from the research drivers; the shared module retains their transitive helpers
so the executed recurrence is unchanged. Original file hashes and extraction
names are in `experiments/affine_gpu/source_manifest.json`.

The small sensitivity uses the original near-equal Ritz-value rule:
`abs(lambda_i-lambda_j) <= 1e-8*max(max(abs(lambda)),1)` selects the derivative
at `lambda_i`. This can produce an asymmetric divided-difference array in
finite precision; the release does not silently symmetrize it. The optional
Ritz clipping setting is exposed explicitly. Chebyshev differentiation freezes
its interval and coefficients in the mass/scale directions; a separate series
computes the alpha derivative. Chebyshev processes the two samples as a batch,
while each Lanczos method processes them sequentially.

Timing covers approximate data-quadratic value and gradient evaluation,
including small-matrix processing, transfers and synchronization. It excludes
spectral reference diagnostics, the shared exact log determinant and prior,
and Adam updates. The Lanczos timer synchronizes each sample; the Chebyshev
path completes its required GPU outputs through blocking CPU reads. There is
no explicit warmup. The saved timings are single runs; trajectory means include
all nine evaluations and are not confidence intervals.

The reported gradient error divides by the full objective gradient norm, clamped
below at 1. Table 8 also reports the error relative to the data-term gradient.
The gradients named `grad_mass` and `grad_scale` in the CSVs are derivatives
with respect to `log(mu)` and `log(s)`. Table 7 is an analytical payload model,
not a measured memory peak: it counts a float32 basis, separately retained
float32 growing stacks, and one float64 small tensor. The stack formula is
`2*n*m*(m+1)` bytes and is conditional on that allocation strategy.

Table 6(b) in the paper truncates its three-decimal parameter display
(e.g. 1.172942 is printed as 1.172). The generated `table_6b.tex` retains that
published display convention. `table_6b_rounded.tex` uses ordinary rounding
(1.173 in the example), and the CSV files retain full precision. The distinction
is display-only; all plots and checks use the original recorded numbers.

The release was additionally validated with one complete reconstructed m128 MPS
run (dimension 32768, all five methods, nine evaluations each). The fresh
projected maximum scaled gradient error was 3.7398e-6, compared with 3.7491e-6
in the archive. Its final scale differed from the archived projected endpoint
by 5.26e-7 and from the spectral-reference endpoint by 6.91e-6 in relative terms.
All 45 rows were finite, and independent recomputation matched every reference
and Adam update exactly. See `results/validation/affine_m128.json` and its raw CSV.
The new timings were collected while other validation jobs ran, and are not
used to establish speedups or reproduce historical timing values.

After running the m128 configuration, the same verification is available as:

```sh
python -m experiments.affine_gpu.validate_m128
```

Use `--results path/to/m128.csv` to inspect a different completed run and
`--output path/to/report-directory` to select the report destination. The
corresponding `.metadata.json` must be present alongside the CSV.
