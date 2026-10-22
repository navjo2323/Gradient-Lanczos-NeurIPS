#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Encoding:
#   Left panel: forward errors only. Use colors distinct from the right panel.
#   Right panel: method encoded by color (Ours=blue, Adjoint=orange).
#   Reorthogonalization encoded consistently by marker/linestyle:
#       full reorth. = solid line + circle marker
#       no reorth.   = dashed line + triangle-down marker

REORTH_STYLE = {
    "full": {"label": "full reorth.", "linestyle": "-", "marker": "o"},
    "none": {"label": "no reorth.", "linestyle": "--", "marker": "v"},
}

# Distinct from blue/orange used for methods on the right.
FORWARD_COLORS = {
    "full": "C2",   # green
    "none": "C4",   # purple
}

METHOD_COLORS = {
    "ours": "C0",     # blue
    "adjoint": "C1",  # orange
}

METHOD_LABELS = {
    "ours": "Ours",
    "adjoint": "Adjoint",
}


def _get(d: dict[str, Any], path: list[str], default: Any = math.nan) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def _positive_or_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y[~np.isfinite(y)] = np.nan
    y[y <= 0] = np.nan
    return y


def _series(rows: list[dict[str, Any]], getter) -> np.ndarray:
    return _positive_or_nan(np.asarray([_float_or_nan(getter(r)) for r in rows], dtype=float))


def plot_errors(payload: dict[str, Any], out_png: Path) -> None:
    rows = payload["rows"]
    ms = np.asarray([_float_or_nan(r.get("m")) for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)

    # Left: forward errors. Colors are not reused by the right panel's method colors.
    for mode in ["full", "none"]:
        st = REORTH_STYLE[mode]
        y = _series(rows, lambda r, mode=mode: _get(r, ["forward", mode, "rel_error"]))
        if np.all(np.isnan(y)):
            continue
        ax0.semilogy(
            ms,
            y,
            color=FORWARD_COLORS[mode],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.2,
            markersize=5.5,
            label=f"Forward: {st['label']}",
        )

    # Right: gradient errors. Color is method; style/marker is reorthogonalization mode.
    for method in ["ours", "adjoint"]:
        for mode in ["full", "none"]:
            st = REORTH_STYLE[mode]
            key = f"{method}_{mode}"
            y = _series(rows, lambda r, key=key: _get(r, ["gradients", key, "rel_error_l2"]))
            if np.all(np.isnan(y)):
                continue
            ax1.semilogy(
                ms,
                y,
                color=METHOD_COLORS[method],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=2.2,
                markersize=5.5,
                label=f"{METHOD_LABELS[method]}, {st['label']}",
            )

    ax0.set_title(r"Forward error: $u^\top \log(K(\theta))u$")
    ax0.set_xlabel("Lanczos iterations $m$")
    ax0.set_ylabel("Relative forward error")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.set_title(r"Gradient error: $\nabla_\theta u^\top \log(K(\theta))u$")
    ax1.set_xlabel("Lanczos iterations $m$")
    ax1.set_ylabel("Relative gradient error")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot forward and gradient error comparison with final paper styling.")
    parser.add_argument("json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.json.read_text())
    out = args.out or args.json.with_name(args.json.stem + "_gradient_comparison_notitle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_errors(payload, out)
    print(json.dumps({"saved_plot": str(out)}, indent=2))


if __name__ == "__main__":
    main()
