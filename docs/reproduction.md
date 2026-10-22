# Reproducing the paper

## What is frozen and what is regenerated

`results/paper/` contains the recovered numerical records used in the final
paper. `scripts/reproduce_paper.py` builds all figures and tables from those
records. It does not estimate new timings. `SHA256SUMS.json` fixes the record
contents; `scripts/verify_release.py` checks the inventory and generated outputs.

`configs/` and the experiment drivers rerun computations. Outputs are written to
ignored output directories, preserving archived records. `scripts/run_experiment.py`
records the chosen configuration, package versions, platform, thread variables,
command and exit code under `outputs/run_records/`. Individual GPU and graph
runners record additional metadata.

A plotted curve can be reproduced from the stored observations even when the
original hardware or some launch flags are unavailable. These are distinct
levels of reproducibility; each experiment guide describes which checks ran.
Timing comparisons on different systems are fresh measurements, not expected
matches to archived seconds. Numerical results may also depend on BLAS, library
versions, reduction order and device precision.

## Experiment commands

Run all commands from the repository root after installation. See the experiment
guides for dataset preparation and exact meanings of settings.

```bash
# Full matrix-log experiment (dense n=15,000 oracle and complete depth sweep)
python scripts/run_experiment.py configs/matrix_log_paper.json

# Protein: pinned download, initialization diagnostic and 120-epoch training
python experiments/gp_training/data.py
python scripts/run_experiment.py configs/gp_diagnostic_paper.json
python scripts/run_experiment.py configs/gp_training_paper.json

# SNAP graphs: download/verify, then all accuracy and complete timing cases
python -m experiments.graph.data
python scripts/run_experiment.py configs/graph_paper.json

# L=8 Hamiltonian: full 400-update dense and projected trajectories
python scripts/run_experiment.py configs/hamiltonian_paper.json

# Affine GPU: reconstructed explicit settings (requires MPS by default)
python scripts/run_experiment.py configs/affine_gpu_m384_reconstructed.json
```

Other affine configs cover depths 128, 768 and 1536. Unknown historical settings
are listed in `docs/affine_gpu.md`; no missing OOM measurement is fabricated.
The CPU quick config exercises the same code with a small well-conditioned
problem. CUDA execution is implemented but has not been validated for this paper.

## Scientific conventions preserved in the release

- The graph method uses two independent scalar Lanczos recurrences in a two-column
  batch, with normalized starts proportional to `e_i+e_j` and `e_i-e_j`. The
  complete-call runner includes setup, projected matrix-function calculations,
  contractions and final assembly. Archived internal timers in accuracy CSVs
  are excluded from the corrected speedup plots and timing tables.
- Both stored graph error columns use `abs(estimate-reference)/max(1,abs(reference))`.
  The release states this normalization explicitly, including for sensitivities.
  The selected value is the matrix-exponential entry for both graph objectives.
- Hamiltonian training uses dense forward propagation for both methods, and the
  baseline uses dense matrix-exponential Fréchet derivatives. The projected
  gradient uses complex polarization into real sine/cosine quadratic forms.
- GP training uses the recovered standardization, split, probe and preconditioner
  protocol. The methods share raw random draws in the diagnostic; their effective
  preconditioned probes differ. See the GP guide for the derivative convention.
- The affine GPU timings cover the data quadratic and its gradient, excluding
  shared exact terms and optimizer updates. The memory table is an analytical
  allocation model, not measured peak memory. Historical GPU flags and software
  versions are not silently inferred from current defaults.

The matrix-log raw source record contains non-standard JSON `NaN` tokens at failed
adjoint points. The original bytes are retained in `results.original.json.raw`;
`results.json` uses JSON `null` for those same points. Plotting preserves failed
points as missing observations.

## Validation and practical limits

`results/validation/` reports the fresh environment checks. Tests compare the
algorithms against independent dense Fréchet derivatives, finite differences,
spectral references and recorded numerical outputs. Small optimization runs
exercise complete training and plotting paths. Full graph accuracy/reference
checks, the Protein diagnostic/training and Hamiltonian initialization checks
provide direct matches to the archived experiments.

Full 400-update Hamiltonian retraining and the full dense n=15,000 matrix-log sweep
are expensive and are not required by the default test suite. Historical GPU
unstable-AD trajectories cannot be promised exactly from incomplete launch
metadata. The precise release checks and any partial reruns are recorded in
`docs/validation.md` and the validation JSON files.

GitHub Actions runs CPU numerical tests, the stored/replay example, all saved-data
figure/table generators and inventory checks. It does not run the expensive full
campaigns or require external datasets. Local validation is not evidence that a
remote CI job has completed; inspect the PR check status separately.
