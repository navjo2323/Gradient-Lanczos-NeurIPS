# Release validation — 22 October 2026

Validation used a newly installed Python 3.11 environment on macOS arm64. Exact
package versions are in `requirements/tested-macos-arm64-py311.txt`. These are
observed release-test versions, not a claim about unrecorded original versions.

## Automated checks

- 29 tests pass: dense Fréchet and finite-difference oracles; stored/replay
  equivalence; GP projected sensitivities and differentiable preconditioning;
  matched matrix-log forwards; graph TN/SC/polarization/reference checks;
  Hamiltonian derivatives; affine spectral references; CLI option handling.
- `python examples/quickstart.py` passes for stored and replay modes.
- `python scripts/reproduce_paper.py` regenerates all eight figures and eight
  tables (41 output files plus its manifest). All figure families were visually
  inspected for legibility and clipping.
- The same plotting command succeeds with Torch, JAX, Matfree, GPyTorch and
  linear_operator imports blocked, confirming that saved-data plotting only
  needs the base dependencies.
- `python scripts/verify_release.py` checks 45 frozen-record hashes, the full
  figure/table mapping and 36 directly mapped output files.
- Python compilation and `pip check` pass. Recovered source files and extracted
  numerical definitions were checked against their original files; source
  manifests document portability changes and JSON serialization changes.

A separate clean Git clone of the committed release passed all 29 tests, the
quickstart, figure/table regeneration and hash/inventory checks. The installed
core module resolved to that clean checkout; no source-collection paths or data
caches were needed. See `results/validation/clean_checkout.json`.

## Direct experiment checks

| Experiment | Fresh computation | Comparison |
|---|---|---|
| Graphs | All 16 graph/objective/depth accuracy cases; four newly computed sparse-block references; eight complete timing cases | All saved numerical values, gradients and references match exactly; new times are fresh measurements |
| Protein GP | Full initialization diagnostic and both 120-epoch training runs | Logdet/gradient diagnostics, all recorded loss and RMSE histories, and final RMSEs match exactly |
| Matrix log | n=15,000, depths 60 and 80, projected/adjoint methods with and without reorthogonalization | All four cases reproduce archived forward values and projected/adjoint gradient coordinates exactly, using the saved dense reference |
| Hamiltonian | Full L=8, 30-sample initialization at depths 30/40/50; small complete optimization | All three recorded initialization gradient errors match exactly |
| Affine GPU records | Independent spectral reference, metric and Adam-update recomputation for all 90 saved rows | All recomputed quantities agree exactly |
| Affine GPU execution | Small CPU/MPS runs and complete n=32,768, m=128, 8-update MPS run across all five methods | All 45 full-run evaluations finite; projected max scaled gradient error 3.740e-6 vs archived 3.749e-6; projected final scale differs by 5.26e-7 |

Detailed reports and the fresh m=128 GPU CSV are in `results/validation/`.
The graph timing rerun and new GPU timing observations occurred alongside other
validation work; they do not replace the paper's controlled timing records.

## Dependency sensitivity found and resolved

A fresh install of JAX/jaxlib 0.10.2 and Matfree 0.6.3 changed the large
unreorthogonalized matrix-log recurrence and made one selected adjoint gradient
nonfinite. Small numerical tests alone did not reveal this difference.
JAX/jaxlib 0.10.0 with Matfree 0.5.5 reproduced the selected archived coordinates,
so those versions are explicitly pinned in the release. Both the comparison and
the final checks are retained in `results/validation/gp_matrix_log.json`.
Pins are based on observed reproduction, not inferred historical metadata.

## Scope and remaining limitations

The entire dense n=15,000 matrix-log reference/depth sweep and full 400-update
Hamiltonian training were not rerun during release packaging. Their archived
records, configurations, scripts and plot/table paths are included. Selected
paper-size matrix-log checks reuse the archived dense reference and therefore
are not a newly computed dense oracle at that size.

The large GPU depth-384/768/1536 campaigns and reported OOM were not rerun. The
historical launch metadata gaps remain explicit. The completed m=128 GPU rerun
uses reconstructed choices and a current PyTorch stack; its small numerical
differences and timings must not be substituted silently for historical results.

The pinned Protein download succeeded and matched its SHA-256. The official SNAP
graph endpoints refused a connection from this environment, even with network
access enabled. Existing local graph files matched their expected SHA-256 values
and were used for the full checks. The download/checksum utility and local-file
fallback are documented; fresh external graph downloads remain unverified here.

The CI workflow is included for repeatable CPU tests and figure regeneration.
Local success does not establish remote CI success; check the pull request's
actual workflow status after publication.
