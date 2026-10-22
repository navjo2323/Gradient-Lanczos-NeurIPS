"""Regenerate graph Figures 3–5 and Tables 1/4 from frozen or fresh records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from ._bootstrap import ROOT

LABELS = {"hepth": "HepTh", "amazon": "Amazon"}
PANELS = [("hepth", "tn"), ("amazon", "tn"), ("hepth", "sc"), ("amazon", "sc")]
TIMING_ORDER = [("hepth", "tn"), ("hepth", "sc"), ("amazon", "tn"), ("amazon", "sc")]
# Transcription of the coarse labels printed in final paper Table 1. These are
# retained as labels, not treated as numerical records or a rounding algorithm.
PAPER_ERROR_EXPONENTS = {
    ("hepth", "tn"): [-4, -4, -14, -11],
    ("amazon", "tn"): [-7, -5, -14, -12],
    ("hepth", "sc"): [-1, -2, -10, -12],
    ("amazon", "sc"): [-8, -8, -14, -15],
}
PAPER_TIMING_DISPLAY = [
    [5.11, 7.01, 1.37, 0.13], [9.81, 14.14, 1.44, 0.16],
    [5.06, 7.09, 1.40, 0.07], [9.95, 14.65, 1.47, 0.03],
    [165.02, 375.70, 2.28, 0.03], [329.80, 884.02, 2.68, 0.03],
    [169.88, 371.73, 2.19, 0.14], [332.19, 892.83, 2.69, 0.09],
]


def timing_summary(record):
    rows = []
    for case in record["cases"]:
        repeats = case["repeats"]
        ours = np.array([r["ours"]["complete_seconds"] for r in repeats])
        ko = np.array([r["ko_seconds"] for r in repeats])
        ratios = ko / ours
        n = len(ratios)
        if n < 2:
            raise ValueError("At least two paired repetitions are needed for a Student-t interval")
        rows.append({"graph": case["graph"], "objective": case["objective"], "depth": case["depth"],
                     "repeats": n, "ours_mean_ms": float(1000 * ours.mean()),
                     "ko_mean_ms": float(1000 * ko.mean()),
                     "mean_paired_speedup": float(ratios.mean()),
                     "ci95_halfwidth": float(t.ppf(0.975, n - 1) * ratios.std(ddof=1) / np.sqrt(n))})
    return rows


def paper_accuracy():
    rows = []
    for graph, objective in PANELS:
        path = ROOT / "results/paper/graph/accuracy" / f"results_paper_batched_safe_{graph}_{objective}.csv"
        with path.open() as handle:
            for row in csv.DictReader(handle):
                rows.append({"graph": graph, "objective": objective, "depth": int(row["k"]),
                             "ours_value_error": float(row["ours_value_err"]),
                             "ko_value_error": float(row["ma_value_err"]),
                             "ours_gradient_error": float(row["ours_grad_err"]),
                             "ko_gradient_error": float(row["ma_grad_err"])})
    return rows


def save_figure(fig, output):
    for extension in ["png", "pdf"]:
        fig.savefig(output.with_suffix("." + extension), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_timing(rows, directory):
    lookup = {(r["graph"], r["objective"], r["depth"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(4)
    for position, depth in enumerate([16, 32]):
        selected = [lookup[(graph, objective, depth)] for graph, objective in TIMING_ORDER]
        bars = ax.bar(x + (position - 0.5) * 0.36,
                      [r["mean_paired_speedup"] for r in selected], 0.36,
                      yerr=[r["ci95_halfwidth"] for r in selected], capsize=3, label=f"$m={depth}$")
        ax.bar_label(bars, labels=[f"{r['mean_paired_speedup']:.2f}" for r in selected], padding=7, fontsize=9)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="0.4")
    ax.set_xticks(x, [f"{LABELS[g]}\n{o.upper()}" for g, o in TIMING_ORDER])
    ax.set_ylabel(r"Runtime ratio $t_{\rm KO}/t_{\rm ours}$")
    ax.set_ylim(0, max(r["mean_paired_speedup"] + r["ci95_halfwidth"] for r in rows) * 1.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, directory / "figure3_graph_speedup")


def plot_accuracy(rows, metric, directory):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True, sharey=True)
    for ax, (graph, objective) in zip(axes.ravel(), PANELS):
        subset = sorted([r for r in rows if r["graph"] == graph and r["objective"] == objective], key=lambda r: r["depth"])
        for method, label, marker, linestyle in [("ours", "Ours", "o", "-"), ("ko", "KO", "s", "--")]:
            ax.plot([r["depth"] for r in subset], [r[f"{method}_{metric}_error"] for r in subset],
                    marker=marker, linewidth=1.9, linestyle=linestyle, label=label)
        ax.set_title(f"{LABELS[graph]} {objective.upper()}")
        ax.set_yscale("log"); ax.set_xticks([4, 8, 16, 32]); ax.grid(True, which="both", axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("Krylov depth $m$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Scaled sensitivity error" if metric == "gradient" else "Scaled exponential-entry error")
    fig.tight_layout()
    save_figure(fig, directory / ("figure4_graph_sensitivity" if metric == "gradient" else "figure5_graph_value"))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def scientific(value):
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def write_tables(timings, accuracy, directory, paper):
    lookup = {(r["graph"], r["objective"], r["depth"]): r for r in timings}
    acc_lookup = {(r["graph"], r["objective"], r["depth"]): r for r in accuracy}
    table4 = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Graph & Obj. & $m$ & Ours (ms) & KO (ms) & Speedup \\", r"\midrule"]
    for graph, objective in TIMING_ORDER:
        for depth in [16, 32]:
            r = lookup[(graph, objective, depth)]
            table4.append(rf"{LABELS[graph]} & {objective.upper()} & {depth} & {r['ours_mean_ms']:.2f} & {r['ko_mean_ms']:.2f} & ${r['mean_paired_speedup']:.2f}\pm{r['ci95_halfwidth']:.2f}$ \\")
    table4 += [r"\bottomrule", r"\end{tabular}"]
    (directory / "table4_graph_timings.tex").write_text("\n".join(table4) + "\n")
    speed = [r"\begin{tabular}{llrr}", r"\toprule", r"Graph & Obj. & $m=16$ & $m=32$ \\", r"\midrule"]
    errors = [r"\begin{tabular}{llrrrr}", r"\toprule", r"& & \multicolumn{2}{c}{$m=16$} & \multicolumn{2}{c}{$m=32$} \\", r"Graph & Obj. & Ours & KO & Ours & KO \\", r"\midrule"]
    labels = errors.copy()
    exact_rows = []
    for graph, objective in PANELS:
        speed.append(rf"{LABELS[graph]} & {objective.upper()} & {lookup[(graph, objective, 16)]['mean_paired_speedup']:.2f} & {lookup[(graph, objective, 32)]['mean_paired_speedup']:.2f} \\")
        values = [acc_lookup[(graph, objective, depth)][f"{method}_gradient_error"] for depth in [16, 32] for method in ["ours", "ko"]]
        errors.append(f"{LABELS[graph]} & {objective.upper()} & " + " & ".join(scientific(value) for value in values) + r" \\")
        labels.append(f"{LABELS[graph]} & {objective.upper()} & " + " & ".join(rf"$10^{{{e}}}$" for e in PAPER_ERROR_EXPONENTS[(graph, objective)]) + r" \\")
        exact_rows.append(dict(zip(["graph", "objective", "ours_m16", "ko_m16", "ours_m32", "ko_m32"], [graph, objective] + values)))
    for lines, name in [(speed, "table1_speedups"), (errors, "table1_sensitivity_errors")]:
        (directory / (name + ".tex")).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    if paper:
        (directory / "table1_paper_coarse_error_labels.tex").write_text(
            "% Labels transcribed from paper Table 1; see exact errors in table1_sensitivity_errors.csv.\n" +
            "\n".join(labels + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    write_csv(directory / "table1_sensitivity_errors.csv", exact_rows)
    write_csv(directory / "table4_graph_timings.csv", timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/generated/graph/plots")
    parser.add_argument("--generated-dir", type=Path, help="Use fresh accuracy.csv and timings.json instead of frozen paper data.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.generated_dir:
        record = json.loads((args.generated_dir / "timings.json").read_text())
        with (args.generated_dir / "accuracy.csv").open() as handle:
            accuracy = [{key: value if key in ["graph", "objective"] else int(value) if key == "depth" else float(value)
                         for key, value in row.items()} for row in csv.DictReader(handle)]
    else:
        record = json.loads((ROOT / "results/paper/graph/timing_source/fresh_timing_results.json").read_text())
        accuracy = paper_accuracy()
    timings = timing_summary(record)
    if not args.generated_dir:
        displayed = [[round(r[key], 2) for key in ["ours_mean_ms", "ko_mean_ms", "mean_paired_speedup", "ci95_halfwidth"]] for r in timings]
        if displayed != PAPER_TIMING_DISPLAY:
            raise AssertionError("Computed timing table differs from the final paper Table 4")
    plot_timing(timings, args.out_dir)
    for metric in ["gradient", "value"]:
        plot_accuracy(accuracy, metric, args.out_dir)
    write_tables(timings, accuracy, args.out_dir, paper=not bool(args.generated_dir))
    write_csv(args.out_dir / "accuracy_plot_data.csv", accuracy)
    report = {"source": "fresh run" if args.generated_dir else "frozen paper records",
              "timing_cases": len(timings), "accuracy_cases": len(accuracy),
              "paper_table4_display_match": None if args.generated_dir else True,
              "error_definition": "abs(estimate-reference)/max(1,abs(reference)) for both value and sensitivity",
              "figures": ["figure3_graph_speedup", "figure4_graph_sensitivity", "figure5_graph_value"]}
    (args.out_dir / "plot_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
