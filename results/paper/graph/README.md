# Graph paper records

See [the reproduction guide](../../../docs/graph.md) for methods, commands and figure/table mappings.

- `accuracy/`: eight unchanged historical CSVs; four all-depth records for Figures 4/5 and four three-repeat records at depths 16/32. Their old runtime columns omit work and must not be used for speedup comparisons.
- `timing_source/fresh_timing_results.json`: authoritative corrected outer-call timing record used by final Figure 3 and Tables 1/4, with three measured pairs per case.
- `timing_source/audit_graph_timing.py`: unchanged historical audit source; portable release entry point is `python -m experiments.graph.run`.
- `graph_timings_raw.csv`, `graph_timings_summary.csv`, `graph_recorded_references.csv`, `run_metadata.json`: unchanged audit exports.
- `regenerated/`: a release-generated snapshot of the graph figures and table exports.
- `outputs/paper/graph/` (at the repository root, after regeneration): figures, TeX fragments, exact table data and plot validation report from the frozen records.
- `../../validation/graph/`: new release checks and recomputed results, stored separately from paper measurements. `validation_summary.json` gives the scope and limitations.
- `provenance.json`: original source hashes, destinations and changes.

Both the original sensitivity and value error fields use `abs(estimate-reference)/max(1,abs(reference))`. Figure axes in the release state this scaled-error convention explicitly. The final paper's Table 1 coarse powers of ten are retained separately from the exact numerical error table.
