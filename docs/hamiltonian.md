# Hamiltonian learning: Table 2 and Figure 6

Run the following commands from the repository root after the README setup.
The base dependencies (`numpy`, `scipy`, `matplotlib`) are sufficient.

```sh
python -m experiments.hamiltonian.plot
python -m experiments.hamiltonian.run --config configs/hamiltonian_quick.json
python -m experiments.hamiltonian.verify_records
python -m experiments.hamiltonian.run --config configs/hamiltonian_paper.json
```

The first command rebuilds `artifacts/hamiltonian/figure_6.{pdf,png}` and
`table_2.{csv,tex}` from the archived `results/paper/hamiltonian/results.json`.
The quick command runs three Adam updates for a three-site, two-sample problem;
it is a functionality check rather than a paper result. The verification command
recreates the full eight-site, 30-sample dataset and recomputes the initial dense
and projected gradients at depths 30, 40 and 50. It writes comparison diagnostics
under `artifacts/hamiltonian/`. The last command retrains all four methods for
400 updates and saves new results under `results/reruns/hamiltonian/`; allow
substantial CPU time. The original dense evaluation history alone took about
73 minutes, excluding dataset construction and diagnostics. Runtime varies by
machine and BLAS implementation.

The `--out-dir` option on the run wrapper selects another destination.
`--results` and `--output` select inputs and outputs for the plotting and
verification commands. New runs include `run_metadata.json` with the actual
configuration, Python/NumPy/SciPy versions, platform and timestamps.

The model is an open-chain Ising Hamiltonian with eight `X_i`, eight `Z_i`,
and seven `Z_i Z_(i+1)` terms. This gives a 256-dimensional real symmetric
matrix and 23 unknown coefficients. Seed 0 generates the ground-truth parameters
and normalized real Gaussian starting states, and uniform observation times
between 2 and 6. The target states use dense `scipy.linalg.expm`. Initialization
uses seed 999 and a Gaussian coefficient perturbation of scale 0.1. Both methods
use this same initialization and 400 Adam updates with learning rate 0.01.
`configs/hamiltonian_paper.json` records every original command-line setting.

Both methods compute the forward residual with a dense matrix exponential.
The dense gradient uses `scipy.linalg.expm_frechet`; the projected gradient uses
four-term complex polarization, real/imaginary splitting, and cosine/sine
sensitivities. Each sample builds 16 independent scalar Lanczos recurrences,
including repeated starts; the historical implementation does not cache them.
The projected matrix sensitivity is shared across all 23 parameter contractions.
The recurrence uses float64 without reorthogonalization. Its historical handling
of a sub-1e-14 residual continues with a zero vector. These numerical choices are
preserved in the byte-for-byte copied `experiments/hamiltonian/train.py`.

The initial gradient error is normalized by `max(norm(g_dense), 1e-14)`.
Parameter error is the absolute Euclidean distance from the true coefficients.
Each stored loss is evaluated before that row's Adam update, while the stored
parameter error is evaluated after the update. The plot preserves this original
indexing. Times accumulate loss-and-gradient evaluation, include dense forward
propagation for both methods, and exclude Adam updates. Table 2 speedups divide
the dense accumulated time by the corresponding projected accumulated time.
Each is one 400-update history, with no independent timing repetitions.

The saved data reproduce the numerical contents of the paper. The release
Figure 6 uses a newly generated two-panel layout with the same histories.
`source_manifest.json` identifies the original source and result hashes.
Historical dependency versions were not saved, so release validation versions
are recorded separately in `results/validation/hamiltonian_gpu.json`. On that
validated environment, all three full-problem initial gradient errors exactly
match the saved floats. Other BLAS/library versions can change an unreorthogonalized
recurrence; verification reports both the new errors and whether they retain
the paper's three-significant-digit values.
