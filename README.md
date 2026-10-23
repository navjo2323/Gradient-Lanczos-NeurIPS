# Gradient through Lanczos — NeurIPS 2026

Code and recorded results for **Fast and Stable Gradient Approximation for
Quadratic Forms of Hermitian Matrix Functions**, by Navjot Singh, Kipton Barros,
and Xiaoye Sherry Li.

This repository contains the experiments used in the paper, their numerical
records, plotting and table scripts, and a small reference implementation of
stored-basis and blockwise replay gradients. The five experiment directories are
self-contained within this repository and its documented Python dependencies.

## Install and run a small example

Python 3.11 is the tested release environment. Run commands from the repository
root. The initial installation needs internet access.

```bash
git clone https://github.com/navjo2323/Gradient-Lanczos-NeurIPS.git
cd Gradient-Lanczos-NeurIPS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/tested-macos-arm64-py311.txt
python -m pip install --no-deps -e .
python examples/quickstart.py
python -m pytest -q
```

The example compares stored-basis and replay gradients with dense SciPy Fréchet
derivatives. See [the algorithm API](docs/algorithm.md) for operator/VJP callbacks
and replay's storage/computation tradeoff. The package also accepts
`pip install -e '.[all]'`; the version file above fixes the environment actually
tested for this release. For only the NumPy/SciPy examples and saved-result plots,
`pip install -e '.[test]'` suffices. GP, matrix-log and GPU tests require their
optional dependencies; use `.[all]` for the entire test suite.

## Recreate every paper figure and table

This command uses the checked-in numerical records, needs no dataset download,
and writes PNG/PDF figures plus CSV/TeX tables to `outputs/paper/`:

```bash
python scripts/reproduce_paper.py
python scripts/verify_release.py
```

It does not repeat expensive experiments or measure new timings. Table scripts
retain the paper's display conventions and provide full-precision CSVs. See
[reproduction details](docs/reproduction.md) for the distinctions between archived
results, new numerical checks, and fresh measurements.

| Paper item | Experiment | Instructions |
|---|---|---|
| Figure 1 | Matrix-log quadratic form and gradients, n=15,000 | [Matrix log](docs/matrix_log.md) |
| Figure 2, Table 3 | Protein GP training and initialization diagnostic | [GP training](docs/gp_training.md) |
| Figures 3–5, Tables 1 and 4 | Graph sensitivity, value accuracy and corrected complete-call timings | [Graphs](docs/graph.md) |
| Figure 6, Table 2 | Hamiltonian learning | [Hamiltonian](docs/hamiltonian.md) |
| Figures 7–8, Tables 5–8 | Affine GPU experiment | [Affine GPU](docs/affine_gpu.md) |

The machine-readable [paper manifest](docs/paper_manifest.json) maps every figure
and table to its records, generator and output. Source manifests record the
original files and checksums before extraction.

## Rerun experiments

Start with small, offline configurations:

```bash
python scripts/run_experiment.py configs/matrix_log_quick.json
python scripts/run_experiment.py configs/gp_diagnostic_quick.json
python scripts/run_experiment.py configs/gp_training_quick.json
python -m experiments.graph.run --mode smoke
python scripts/run_experiment.py configs/hamiltonian_quick.json
python scripts/run_experiment.py configs/affine_gpu_quick.json
```

Each experiment guide documents full configurations, dataset download/checksum
commands, numerical conventions, output locations and verification. Full runs can
require substantial memory or runtime. The GPU study's historical launch settings
are only partly recoverable; the provided rerun configs explicitly record their
reconstructed choices. Archived GPU figures/tables can be regenerated exactly
from the saved numerical records, while matching historical unstable AD behavior
or timings on a new software stack is not guaranteed.

## Organization

```text
src/lanczos_gradients/   small reference API: stored basis and blockwise replay
examples/               dense-oracle quickstart
experiments/            the five paper experiment implementations and plotting code
configs/                small and paper-scale configurations
results/paper/          frozen paper records and provenance
results/validation/     release validation reports
scripts/                configuration runner, all-paper regeneration, inventory checks
tests/                  independent numerical oracles and archived-result checks
docs/                   reproduction guides and figure/table manifest
third_party/            upstream baseline code and license notices
requirements/           exact release-test dependency versions
```

## Citation and licenses

Use [CITATION.cff](CITATION.cff) to cite the paper and software. Original code is
MIT licensed. The modified-Arnoldi graph baseline retains GPLv2, and upstream
experiment code retains its MIT notice. See [third-party notices](THIRD_PARTY_NOTICES.md).
Dataset licenses are separate from the code license; the datasets are downloaded
from the documented sources and verified by checksum.
