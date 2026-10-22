"""Scientific checks of the graph protocol and frozen paper artifacts."""
import csv
import hashlib
import json

import numpy as np
import pytest

from experiments.graph._bootstrap import ROOT
from experiments.graph.data import verify
from experiments.graph.plots import PAPER_TIMING_DISPLAY, timing_summary
from experiments.graph.run import kernels, smoke
from benchmark_pkg_graph_io import load_graph_matrix


def test_graph_dense_oracle_for_both_directions_and_swapped_start():
    result = smoke()
    assert result["status"] == "passed"
    assert result["checks"][1]["ko_kind"].endswith("swapped_start")


def test_graph_batch_is_two_scalar_recurrences():
    rng = np.random.default_rng(48)
    matrix = rng.normal(size=(20, 20))
    matrix = (matrix + matrix.T) / 4
    starts = np.column_stack([np.eye(20)[2] + np.eye(20)[7], np.eye(20)[2] - np.eye(20)[7]])
    batched = kernels.lanczos_forward_batched(lambda block: matrix @ block, starts, 6)
    np.testing.assert_allclose(batched.norm_v, np.sqrt(2), rtol=1e-15)
    for column in range(2):
        serial = kernels.lanczos_forward(lambda vector: matrix @ vector, starts[:, column], 6)
        np.testing.assert_allclose(batched.a[column], serial.a, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(batched.b[column], serial.b, rtol=1e-12, atol=1e-13)


def test_graph_loader_paper_symmetrization_and_duplicate_convention(tmp_path):
    path = tmp_path / "edges.txt"
    path.write_text("1 5\n1 5\n5 8\n8 8\n")
    actual = load_graph_matrix(str(path)).toarray()
    np.testing.assert_array_equal(actual, [[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]])


def test_graph_data_verification_rejects_wrong_file(tmp_path):
    (tmp_path / "ca-HepTh.txt").write_text("1 2\n")
    with pytest.raises(ValueError, match="Unexpected SHA-256"):
        verify("hepth", tmp_path)


def test_graph_recorded_timing_rounds_to_final_paper_table4():
    record = json.loads((ROOT / "results/paper/graph/timing_source/fresh_timing_results.json").read_text())
    rows = timing_summary(record)
    display = [[round(row[key], 2) for key in ["ours_mean_ms", "ko_mean_ms", "mean_paired_speedup", "ci95_halfwidth"]] for row in rows]
    assert display == PAPER_TIMING_DISPLAY
    with (ROOT / "results/paper/graph/graph_timings_summary.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    for computed, expected in zip(rows, saved, strict=True):
        np.testing.assert_allclose(computed["mean_paired_speedup"], float(expected["speedup_mean_paired_ratios"]), rtol=1e-14)
        # Historical export used a tabulated t critical value; scipy's inverse
        # CDF differs at ~4e-11 relative, far below the displayed precision.
        np.testing.assert_allclose(computed["ci95_halfwidth"], float(expected["speedup_ci95_halfwidth"]), rtol=1e-9)


def test_graph_recorded_accuracy_errors_use_the_actual_normalization():
    paths = list((ROOT / "results/paper/graph/accuracy").glob("*.csv"))
    assert len(paths) == 8
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                for method in ["ours", "ma"]:
                    for quantity in ["value", "grad"]:
                        reference = float(row[f"reference_{quantity}"])
                        computed = abs(float(row[f"{method}_{quantity}"]) - reference) / max(1, abs(reference))
                        np.testing.assert_allclose(computed, float(row[f"{method}_{quantity}_err"]), rtol=1e-14, atol=1e-30)


def test_graph_copied_source_and_raw_data_are_unchanged():
    records = json.loads((ROOT / "results/paper/graph/provenance.json").read_text())["sources"]
    for row in records:
        if row["changes"] == "none":
            assert hashlib.sha256((ROOT / row["destination"]).read_bytes()).hexdigest() == row["original_sha256"]
