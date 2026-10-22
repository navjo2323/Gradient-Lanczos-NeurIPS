# Graph sensitivities: reproduction guide

This experiment reproduces Section 5.2 and Appendix B.2 of *Fast and Stable Gradient Approximation for Quadratic Forms of Hermitian Matrix Functions*: Figures 3–5 and Tables 1 and 4. It includes the original numerical kernels, the corrected complete-call timing runner, all recorded accuracy results and references, and the three paired timing measurements per configuration used in the final paper.

## Install and run a small check

Use the repository's Python environment. The graph experiment requires NumPy, SciPy, Matplotlib and threadpoolctl; pytest runs the checks. It does not require JAX, PyTorch or a GPU.

From the repository root:

```bash
python -m experiments.graph.run --mode smoke
python -m pytest -q tests/test_graph_reproduction.py
```

The smoke example generates a deterministic 12-node weighted symmetric graph (seed 284). For both TN and SC, it checks the projected method, the modified-Arnoldi baseline and the sparse-block reference against dense `scipy.linalg.expm` and `expm_frechet`. It also exercises the SC baseline's start-vector orientation safeguard. No external dataset is needed.

## Recreate paper plots and tables without rerunning experiments

```bash
python -m experiments.graph.plots
```

Outputs go to `results/generated/graph/plots/`. This command reads the frozen records in `results/paper/graph/`, recomputes paired speedups and Student-t intervals, and checks all displayed Table 4 numbers against the final paper. It generates:

| Paper item | Output stem | Input records |
| --- | --- | --- |
| Figure 3 | `figure3_graph_speedup` (PDF and PNG) | `timing_source/fresh_timing_results.json` |
| Figure 4 | `figure4_graph_sensitivity` (PDF and PNG) | Four `accuracy/results_paper_batched_safe_{graph}_{objective}.csv` files |
| Figure 5 | `figure5_graph_value` (PDF and PNG) | Same four accuracy CSVs |
| Table 1 speedups | `table1_speedups.tex` | Complete-call timing records |
| Table 1 sensitivity errors | `table1_sensitivity_errors.tex` and `.csv` | Recorded errors at depths 16 and 32 |
| Table 1 printed coarse labels | `table1_paper_coarse_error_labels.tex` | Explicit transcription of the final paper's powers of ten |
| Table 4 | `table4_graph_timings.tex` and `.csv` | Complete-call timing records |

The TeX fragments use `booktabs`. The exact-error table prints two significant digits and the CSV retains full precision. The paper's Table 1 uses coarse powers of ten that are not a uniform rounding rule; the separately named label fragment preserves those labels without treating them as raw numerical data. Plot coordinates and timing statistics reproduce the paper records; the figures use descriptive axis labels and are not intended to be pixel-identical copies of the manuscript artwork.

## Obtain the exact graphs

```bash
python -m experiments.graph.data
```

The downloader fetches public SNAP gzip files, decompresses them into `data/graphs/`, and verifies the decompressed SHA-256 hashes before writing. Existing files are also verified. Graph data are not bundled in Git.

| Graph | Official source | Decompressed SHA-256 |
| --- | --- | --- |
| ca-HepTh | [SNAP ca-HepTh](https://snap.stanford.edu/data/ca-HepTh.html) | `8ce8a9aa75f0617d574a8109a6f8b189a9e38d214a6698ceaf25b2eecc873e90` |
| com-Amazon | [SNAP Amazon](https://snap.stanford.edu/data/com-Amazon.html) | `9c403688d01417ca452910947a7000bcd61965fc583f4afe57eb209c8b057b7a` |

The SNAP pages supply dataset descriptions and citations: Leskovec, Kleinberg and Faloutsos (2007) for ca-HepTh; Yang and Leskovec (2012) for Amazon. Dataset use follows the terms of the upstream source; these downloads are separate from the MIT-licensed original code.

To use existing local edge lists, pass their directory to both verification and benchmarking:

```bash
python -m experiments.graph.data --data-dir /path/to/graphs --verify-only
python -m experiments.graph.run --mode both --data-dir /path/to/graphs
```

The loader remaps sorted unique node IDs to contiguous zero-based indices, sums duplicate entries, removes self-loops, and forms `(A + A.T) / 2`. This produces 9,877 rows and 51,946 nonzeros for ca-HepTh, and 334,863 rows and 1,851,744 nonzeros for Amazon. In particular, a one-way edge-list entry becomes a symmetric edge of weight 0.5. Changing this loading convention changes the experiment.

## Recompute every graph result

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.graph.run --mode both --reference recompute
python -m experiments.graph.plots --generated-dir results/generated/graph \
  --out-dir results/generated/graph/fresh_plots
```

`configs/graph_paper.json` specifies both graphs, TN and SC, accuracy depths 4/8/16/32, timing depths 16/32, one warm-up and three paired measurements. The complete run computes new sparse-block references, checks each method's new values and gradients against the recorded outputs, measures complete value-and-gradient calls, and saves `accuracy.csv`, `timings.json` and `metadata.json` separately from the frozen paper records. With the default `--reference stored`, it uses the saved sparse-block references instead of recomputing them. `--mode accuracy` and `--mode timing` run the two parts independently; `--graphs hepth` selects only ca-HepTh.

New wall-clock timings depend on the machine, BLAS implementation and load. Reproducing the numerical results does not require new timings to equal the historical timings. Concurrent work can affect measurements; run timing studies on an otherwise idle machine. The runner requests one thread and records the thread pools that threadpoolctl can detect. An empty list does not establish that every underlying library used one thread.

## Numerical protocol and definitions

For each graph, the returned value is the exponential entry `e_i.T @ exp(A) @ e_j`. TN and SC identify the associated sensitivity direction: `E = ones @ ones.T` for TN and `E = e_ell @ e_ell.T` for SC. The gradient is the bilinear Fréchet derivative `e_i.T @ L_exp(A,E) @ e_j`, using the ambient entrywise convention. A derivative with respect to an undirected edge that changes both off-diagonal entries is twice this sensitivity.

The selected indices are `(i,j)=(7200,6969), ell=7200` for ca-HepTh and `(0,53525), ell=0` for Amazon. The raw adjacency has no spectral rescaling; theta is zero and the time scale is one. The projected method uses two independent scalar Lanczos recurrences initialized with `(e_i+e_j)/sqrt(2)` and `(e_i-e_j)/sqrt(2)`. They run together as a two-column batch, in float64, without reorthogonalization, with breakdown tolerance `1e-14`. The quadratic forms include the squared input norm 2 and are combined by one-quarter polarization. This is not block Lanczos.

Both stored error columns use the actual implementation's formula `abs(estimate-reference) / max(1,abs(reference))`. This includes the sensitivity error, even where the manuscript calls it a relative error. The distinction matters for SC, whose reference sensitivity has magnitude below one: ca-HepTh approximately 0.06896 and Amazon approximately 0.80052. The release preserves all recorded error values and labels the regenerated plots as *scaled* errors. Figure 5 measures the selected exponential-entry value, not the TN or SC objective value.

The reference uses `scipy.sparse.linalg.expm_multiply` on the 2n-dimensional Fréchet block operator with library-default tolerances. The modified-Arnoldi baseline is the unmodified `fAb-frechet/lib.py` implementation associated with Kressner and Oehme. For SC, the wrapper swaps the two test vectors when `E @ e_j` is zero and `E @ e_i` is nonzero; symmetry preserves the target scalar and gives a nonzero derivative starting vector.

## Timing and provenance

Use only `timing_source/fresh_timing_results.json`, `graph_timings_raw.csv` and `graph_timings_summary.csv` for the final paper timing claims. The runtime fields in the historical accuracy CSVs cover unequal work and are retained only as part of the unchanged source records. The plot generator does not use them.

The release runner preserves the corrected audit's `evaluate_complete`, `measure_ours` and `measure_ko` bodies. Outer timers cover matrix preparation, both Lanczos recurrences, tridiagonal construction, matrix exponentials and sensitivities, direction contractions, scalar assembly, baseline start-orientation checking and call cleanup. Graph loading, reference calculations and warm-ups are outside the measured regions. The unused exact Lanczos adjoint is excluded. Each case receives one warm-up per method and three measured pairs, alternating method order. Speedup is the mean of the three paired `t_KO/t_ours` ratios, not the ratio of mean runtimes; intervals are mean ± Student-t(0.975, 2) × sample SD / sqrt(3).

The preserved historical timing metadata identify Python 3.11.14, NumPy 2.4.4, SciPy 1.17.1 and macOS 26.2 arm64. The final manuscript identifies Apple M4 Pro hardware, but the original machine-readable timing record did not capture hardware model, RAM or a wall-clock timestamp. Those absent fields remain unknown in `run_metadata.json`. Original accuracy-run package versions were not recorded. Current validation versions are separately recorded in the validation metadata.

`results/paper/graph/provenance.json` gives the source collection, source-relative path, original SHA-256, destination and changes for every copied file. `experiments/graph/implementation/` contains unchanged original modules. Its `paper_benchmark.py` is retained for the orientation helper and provenance; use the release entry points above for corrected timing. `results/paper/graph/timing_source/audit_graph_timing.py` is an unchanged historical source snapshot with its original local path, not the portable entry point.

The separately licensed modified-Arnoldi code and its GPLv2 license are in `third_party/fAb-frechet/`. See that directory's README for attribution. The repository MIT license covers original release code, not that baseline.

## Release validation

`results/validation/graph/` records fresh smoke checks, complete graph accuracy results, timing validation and environment metadata. The release checks compare all 16 graph/objective/depth cases and all four recomputed sparse-block references with the saved records. Preserved source hashes, loader conventions, two independent starting recurrences and all displayed Table 4 values are also checked.

A fresh external SNAP download was attempted during packaging but the connection was refused by the execution environment. The official source URLs were verified and the existing local datasets passed the exact SHA-256 checks. The deterministic unit test verifies rejection of an incorrect dataset file. This distinguishes validation of the local datasets from an uncompleted network transfer.
