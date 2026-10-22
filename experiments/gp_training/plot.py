"""Regenerate Figure 2 and Table 3 from saved experiment JSON files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def plot_training(payload: dict, output: Path) -> None:
    styles = {"gpytorch": ("GPyTorch", "C2", "--"), "ours_no_reorth": ("Ours", "C0", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for method in payload["methods"]:
        label, color, linestyle = styles[method["method"]]
        history = method["history"]
        axes[0].plot([r["iteration"] for r in history], [r["loss"] for r in history],
                     label=label, color=color, linestyle=linestyle, linewidth=2)
        evaluations = [r for r in history if r["rmse"] is not None]
        axes[1].plot([r["iteration"] for r in evaluations], [r["rmse"] for r in evaluations],
                     label=label, color=color, linestyle=linestyle, linewidth=2)
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Train negative log likelihood / n")
    axes[1].set_ylabel("Test RMSE")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def diagnostic_rows(payload: dict) -> list[dict]:
    return [dict(method=label, scaled_logdet_error=payload["methods"][key]["logdet_rel_error_to_dense"],
                 relative_gradient_error=payload["methods"][key]["grad_rel_error_to_dense"])
            for key, label in [("gpytorch_preconditioned", "GPyTorch"), ("ours_no_reorth_preconditioned", "Ours")]]


def _latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def export_table(payload: dict, outdir: Path) -> None:
    rows = diagnostic_rows(payload)
    with (outdir / "table3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Method & Scaled logdet error & Rel. gradient error \\", r"\midrule"]
    lines.extend(f"{r['method']} & {_latex_scientific(r['scaled_logdet_error'])} & {_latex_scientific(r['relative_gradient_error'])} \\\\" for r in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table3.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "results/paper/gp_training/training.json")
    parser.add_argument("--diagnostic", type=Path, default=ROOT / "results/paper/gp_training/diagnostic.json")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/figures/gp_training"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.training.read_text())
    for suffix in ("png", "pdf"):
        plot_training(payload, args.outdir / f"figure2.{suffix}")
    export_table(json.loads(args.diagnostic.read_text()), args.outdir)
    print(args.outdir)


if __name__ == "__main__":
    main()
