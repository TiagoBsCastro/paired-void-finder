"""Plot summaries of a parameter scan produced by run_parameter_scan.py.

Usage
-----
python scripts/plot_parameter_scan.py \\
    --results  results/scan_geometry/param_scan_results.csv \\
    --outdir   results/scan_geometry/plots/

For each scanned parameter and each metric, a line plot is produced showing
how the metric varies when that parameter is swept with all others held at
their most-common (mode) value.  When exactly two parameters were scanned,
2D heatmaps are also produced.

Metrics plotted
---------------
- n_matched          (integer — more is better)
- n_missed           (integer — fewer is better)
- false_positive_count (integer — fewer is better)
- n_duplicate_matches  (integer — fewer is better)
- mean_center_error  (float — smaller |value| is better)
- mean_radius_error  (float — smaller |value| is better)
- mean_volume_error  (float — smaller |value| is better)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot parameter scan results from param_scan_results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", required=True, help="Path to param_scan_results.csv.")
    p.add_argument("--outdir",  required=True, help="Directory to save plots.")
    return p.parse_args()


# ── CSV loading ───────────────────────────────────────────────────────────────

_PARAM_COLS  = ["b_BB", "eta", "b_shell", "b_grow", "lambda_alpha"]
_METRIC_COLS = [
    "n_matched", "n_missed", "false_positive_count", "n_duplicate_matches",
    "mean_center_error", "mean_radius_error", "mean_volume_error",
]


def _load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Return a dict mapping column name → float array (NaN for missing values)."""
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    result: dict[str, np.ndarray] = {}
    for col in _PARAM_COLS + _METRIC_COLS:
        values = []
        for row in rows:
            raw = row.get(col, "")
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                values.append(float("nan"))
        result[col] = np.asarray(values, dtype=float)
    return result


# ── Plotting helpers ──────────────────────────────────────────────────────────


def _scanned_params(data: dict[str, np.ndarray]) -> list[str]:
    """Return the subset of parameter columns that have more than one unique value."""
    return [p for p in _PARAM_COLS if len(np.unique(data[p][~np.isnan(data[p])])) > 1]


def _mode_values(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Most-common value of each parameter column (used to hold others fixed)."""
    mode_vals: dict[str, float] = {}
    for p in _PARAM_COLS:
        vals = data[p][~np.isnan(data[p])]
        if len(vals) == 0:
            mode_vals[p] = float("nan")
            continue
        unique, counts = np.unique(vals, return_counts=True)
        mode_vals[p] = float(unique[np.argmax(counts)])
    return mode_vals


def _line_plots(data, scanned, mode_vals, outdir):
    """One figure per scanned parameter, all metrics in a grid of subplots."""
    n_metrics = len(_METRIC_COLS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    for param in scanned:
        # Select rows where all *other* scanned params are at their mode value.
        mask = np.ones(len(data[param]), dtype=bool)
        for other in scanned:
            if other == param:
                continue
            mask &= np.isclose(data[other], mode_vals[other], atol=1e-9, equal_nan=False)

        x = data[param][mask]
        order = np.argsort(x)
        x = x[order]

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(f"Metrics vs {param}", fontsize=11)

        for mi, metric in enumerate(_METRIC_COLS):
            row, col = divmod(mi, ncols)
            ax = axes[row][col]
            y = data[metric][mask][order]
            valid = ~np.isnan(y)
            ax.plot(x[valid], y[valid], "o-", markersize=4, linewidth=1.2)
            ax.set_xlabel(param, fontsize=8)
            ax.set_ylabel(metric, fontsize=8)
            ax.set_title(metric, fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused subpanels.
        for mi in range(n_metrics, nrows * ncols):
            row, col = divmod(mi, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        outpath = outdir / f"line_{param}.png"
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        print(f"  Saved {outpath.name}")


def _heatmaps(data, scanned, outdir):
    """2D heatmap for each pair of scanned parameters × each metric."""
    if len(scanned) < 2:
        return

    import itertools
    for p1, p2 in itertools.combinations(scanned, 2):
        u1 = np.unique(data[p1][~np.isnan(data[p1])])
        u2 = np.unique(data[p2][~np.isnan(data[p2])])

        n_metrics = len(_METRIC_COLS)
        ncols = 3
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(f"Heatmaps: {p1} × {p2}", fontsize=11)

        for mi, metric in enumerate(_METRIC_COLS):
            row, col = divmod(mi, ncols)
            ax = axes[row][col]

            grid = np.full((len(u1), len(u2)), float("nan"))
            for i, v1 in enumerate(u1):
                for j, v2 in enumerate(u2):
                    mask = np.isclose(data[p1], v1, atol=1e-9) & np.isclose(data[p2], v2,
                                                                               atol=1e-9)
                    vals = data[metric][mask]
                    valid = vals[~np.isnan(vals)]
                    if len(valid):
                        grid[i, j] = float(np.mean(valid))

            im = ax.imshow(grid, origin="lower", aspect="auto",
                           extent=[u2[0], u2[-1], u1[0], u1[-1]])
            plt.colorbar(im, ax=ax, pad=0.02)
            ax.set_xlabel(p2, fontsize=8)
            ax.set_ylabel(p1, fontsize=8)
            ax.set_title(metric, fontsize=8)
            ax.tick_params(labelsize=7)

        for mi in range(n_metrics, nrows * ncols):
            row, col = divmod(mi, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        outpath = outdir / f"heatmap_{p1}_vs_{p2}.png"
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        print(f"  Saved {outpath.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_csv(args.results)
    scanned = _scanned_params(data)
    mode_vals = _mode_values(data)

    print(f"Loaded {len(data[_PARAM_COLS[0]])} rows.")
    print(f"Scanned parameters: {scanned or '(none — single point)'}")
    print(f"Mode values: {mode_vals}")

    if not scanned:
        print("Nothing to plot — all parameters have a single value.")
        return

    _line_plots(data, scanned, mode_vals, outdir)
    _heatmaps(data, scanned, outdir)
    print(f"All plots saved to {outdir}")


if __name__ == "__main__":
    main()
