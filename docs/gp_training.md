# Gaussian-process training: Figure 2 and Table 3

The release includes the original PyTorch GP training and initialization
diagnostic implementations, their exact archived result files, a plotting/table
export script, and explicit paper and quick-run configurations. The standalone
helper contains only the imported functions used by this paper. Its numerical
preprocessing, subset selection, and synthetic data generation are unchanged.

After installing the experiment dependencies, run the offline synthetic examples:

```sh
python scripts/run_experiment.py configs/gp_training_quick.json
python scripts/run_experiment.py configs/gp_diagnostic_quick.json
```

The quick example uses 64 synthetic observations and three Adam iterations. The
initialization diagnostic compares both stochastic methods against a dense
Cholesky value and derivative. These examples check installation and execution;
their values are not the paper's Protein results.

For the full experiment, first download the exact data bytes and verify the hash:

```sh
python experiments/gp_training/data.py
python scripts/run_experiment.py configs/gp_diagnostic_paper.json
python scripts/run_experiment.py configs/gp_training_paper.json
```

The downloader fetches the 1.6 MB compressed Protein file from the
[uci_datasets collection](https://github.com/treforevans/uci_datasets/tree/f0737523712fe66a1640e4b980b927e8490b58eb),
commit `f0737523712fe66a1640e4b980b927e8490b58eb`. Its SHA256 is
`f709d0107c3ee90ab019b9e5d77fc2ae182158958cb70b1903236a51334fd01b`.
The file is cached at `data/protein/data.csv.gz`. Set `LANCZOS_DATA_DIR` to relocate
the cache; its `protein/data.csv.gz` child must contain these exact bytes. The
training loader checks the hash each time. The experiment uses the full arrays
from this collection, not its supplied train/test masks. The collection credits
the UCI repository; see its repository and dataset documentation for attribution.

The paper protocol standardizes features over all 45,730 examples, drops constant
features, and centers/scales the full target array before selecting 4096 examples
with NumPy seed 0. Selected indices are sorted. PyTorch's seed-0 permutation then
splits them into 3276 training and 820 test examples. This preprocessing includes
the eventual test observations in its normalization statistics, as reported in
the paper.

Both methods use a Matérn-3/2 ARD covariance with nine lengthscales, outputscale,
constant mean and a Gaussian likelihood. The code shares the same initial model
state (raw parameter seed 0), then runs 120 Adam updates at learning rate 0.025.
It uses CPU float64, Lanczos depth 40, 20 probes, a rank-15 pivoted-Cholesky
preconditioner, CG tolerance `1e-4`, and evaluates test RMSE every three updates.
All flags are explicit in the configuration files. The archived device resolved
to CPU; the original thread count and library versions were not recorded.

Our method normalizes independent Gaussian probes and weights each quadrature
by the training size. Lanczos acts on `P^(-1/2) K P^(-1/2)`. Its backward
contraction holds the Lanczos basis and projected sensitivity fixed while
differentiating the covariance, the preconditioner inverse square root and
`logdet(P)`. Pivot choices are discrete and fixed during backward evaluation;
selected-row Cholesky calculations are differentiated. The GPyTorch implementation
uses its solve-based backward estimator. In the initialization diagnostic, both
methods receive the same raw Gaussian array, but the effective preconditioned
probes differ. During training, a new seeded probe array is used each iteration.

Regenerate the paper plots and Table 3 from the archived measurements:

```sh
python experiments/gp_training/plot.py
```

This writes `outputs/figures/gp_training/figure2.png`, `figure2.pdf`, `table3.csv`,
and `table3.tex`. The table fragment uses the LaTeX `booktabs` package. To plot a
fresh full run, pass its JSON files through `--training` and `--diagnostic`.
The figure's left panel shows each method's own stochastic estimate of the
negative log marginal likelihood per training observation. The right panel shows
held-out RMSE. Objective values from the two stochastic estimators need not agree.

Archived final test RMSE is `0.7446171847951735` for GPyTorch and
`0.6887978068780842` for the projected method. Table 3 scales logdet error by
`max(1, abs(dense_logdet))`; gradient error is the Euclidean error divided by the
dense gradient norm in raw model-parameter coordinates. The reported errors
include stochastic estimation error. The two table rows round to
`2.29e-3 / 1.44e-1` and `1.94e-3 / 5.10e-3` for scaled value / relative gradient
error, respectively.

`results/paper/gp_training/training.json` and `diagnostic.json` are unmodified
historical records, including their original local output paths. Those path
strings are provenance only; the release does not read or write those locations.
`figure2_original.png` preserves the original figure. Regenerated plots use all
saved measurements with portable formatting. Source hashes and the limited
portability changes are listed in `docs/gp_matrix_log_source_manifest.json`.

Release validation reran the complete 120-update Protein training and full
initialization diagnostic in the clean, pinned environment using the downloaded,
hash-verified dataset. Both methods reproduced every saved loss and evaluated
RMSE exactly; the diagnostic values and error measures also matched. The
machine-readable comparisons, environment versions and commands are recorded in
`results/validation/gp_matrix_log.json`. This equality concerns numerical
measurements; runtime values depend on the machine and concurrent workloads.
