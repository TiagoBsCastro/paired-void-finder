"""Parameter scan over key void-finder parameters for a single mock.

Usage
-----
python scripts/run_parameter_scan.py \\
    --mock-config  configs/mock_geometry.yaml \\
    --finder-config configs/algorithm_default.yaml \\
    --outdir        results/scan_geometry/ \\
    [--b-BB         1.0 1.5 2.0] \\
    [--eta          0.3 0.5 0.7] \\
    [--b-shell      1.0 1.5 2.0] \\
    [--b-grow       0.0 0.5 1.0] \\
    [--lambda-alpha 1.5 2.0 3.0 4.0]

The finder YAML sets the *base* parameters; only the values explicitly
specified on the command line are varied.  Every parameter combination is
evaluated independently (full Cartesian product).

Output
------
``<outdir>/param_scan_results.csv``  — one row per (param combo, matched void)
``<outdir>/param_scan_summary.csv``  — one row per param combo (aggregate stats)

CSV columns (param_scan_results.csv):
    b_BB, eta, b_shell, b_grow, lambda_alpha,
    n_recovered, n_matched, n_missed, n_duplicate_matches, false_positive_count,
    mean_center_error, mean_radius_error, mean_volume_error
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.validation import validate_against_mock
from paired_void_finder.voids import run_void_finder


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid parameter scan for the paired void finder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mock-config",   required=True, help="Mock YAML config file.")
    p.add_argument("--finder-config", required=True, help="Base finder YAML config file.")
    p.add_argument("--outdir",        required=True, help="Output directory.")

    p.add_argument("--b-BB",         nargs="+", type=float, default=None,
                   metavar="V", help="b_BB values to scan.")
    p.add_argument("--eta",          nargs="+", type=float, default=None,
                   metavar="V", help="eta values to scan.")
    p.add_argument("--b-shell",      nargs="+", type=float, default=None,
                   metavar="V", help="b_shell values to scan.")
    p.add_argument("--b-grow",       nargs="+", type=float, default=None,
                   metavar="V", help="b_grow values to scan.")
    p.add_argument("--lambda-alpha", nargs="+", type=float, default=None,
                   metavar="V", help="lambda_alpha values to scan.")
    return p.parse_args()


# ── Mock loading ──────────────────────────────────────────────────────────────


def _load_mock(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    centers = cfg.get("void_centers")
    radii = cfg.get("void_radii")
    if centers is None or radii is None:
        raise ValueError("mock YAML must contain 'void_centers' and 'void_radii'.")

    return make_swiss_cheese_mock(
        box_size=float(cfg.get("box_size", 100.0)),
        n_points=int(cfg.get("n_points", 10000)),
        void_centers=np.asarray(centers, dtype=float),
        void_radii=np.asarray(radii, dtype=float),
        mode=str(cfg.get("mode", "geometry")),
        seed=int(cfg.get("seed", 0)),
        exterior_b_fraction=float(cfg.get("exterior_b_fraction", 0.0)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load mock once — reused for every parameter combination.
    mock = _load_mock(args.mock_config)
    print(
        f"Mock: {len(mock.A.positions)} A, {len(mock.B.positions)} B, "
        f"{len(mock.true_void_radii)} true voids"
    )

    # Load base finder parameters.
    base = FinderParameters.from_yaml(args.finder_config)

    # Build scan axes: use the command-line values when given, otherwise a
    # single-element list containing the base parameter value.
    axes = {
        "b_BB":         args.b_BB         or [base.b_BB],
        "eta":          args.eta          or [base.eta],
        "b_shell":      args.b_shell      or [base.b_shell],
        "b_grow":       args.b_grow       or [base.b_grow],
        "lambda_alpha": args.lambda_alpha or [base.lambda_alpha],
    }

    keys   = list(axes.keys())
    combos = list(itertools.product(*[axes[k] for k in keys]))
    n_total = len(combos)
    print(
        f"Parameter grid: {' × '.join(f'{len(axes[k])} {k}' for k in keys)} "
        f"= {n_total} combinations"
    )

    results_path = outdir / "param_scan_results.csv"
    summary_path = outdir / "param_scan_summary.csv"

    _RESULT_HEADER = [
        "b_BB", "eta", "b_shell", "b_grow", "lambda_alpha",
        "n_recovered", "n_matched", "n_missed", "n_duplicate_matches",
        "false_positive_count",
        "mean_center_error", "mean_radius_error", "mean_volume_error",
    ]

    t0 = time.time()
    with (
        open(results_path, "w", newline="", encoding="utf-8") as rf,
        open(summary_path, "w", newline="", encoding="utf-8") as sf,
    ):
        rw = csv.writer(rf)
        sw = csv.writer(sf)
        rw.writerow(_RESULT_HEADER)
        sw.writerow(_RESULT_HEADER)

        for idx, combo in enumerate(combos, 1):
            vals = dict(zip(keys, combo))
            params = FinderParameters(
                M_A_min=base.M_A_min,
                M_B_min=base.M_B_min,
                b_BB=vals["b_BB"],
                eta=vals["eta"],
                N_veto=base.N_veto,
                boundary_mode=base.boundary_mode,
                b_shell=vals["b_shell"],
                b_grow=vals["b_grow"],
                lambda_alpha=vals["lambda_alpha"],
                N_B_min=base.N_B_min,
                N_A_min=base.N_A_min,
                R_min=base.R_min,
                enable_veto=base.enable_veto,
            )

            try:
                voids = run_void_finder(mock.A, mock.B, params)
                summary = validate_against_mock(voids, mock)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{idx}/{n_total}] combo {vals} FAILED: {exc}")
                _write_error_row(rw, sw, vals, keys)
                continue

            n_rec   = summary.n_recovered
            n_mat   = summary.n_matched
            n_mis   = summary.n_missed
            n_dup   = summary.n_duplicate_matches
            n_fp    = summary.false_positive_count
            ce_mean = float(np.mean(summary.center_errors)) if n_mat > 0 else float("nan")
            re_mean = float(np.mean(summary.radius_errors)) if n_mat > 0 else float("nan")
            ve_mean = float(np.mean(summary.volume_errors)) if n_mat > 0 else float("nan")

            row = [
                vals["b_BB"], vals["eta"], vals["b_shell"], vals["b_grow"], vals["lambda_alpha"],
                n_rec, n_mat, n_mis, n_dup, n_fp,
                f"{ce_mean:.4f}", f"{re_mean:.4f}", f"{ve_mean:.4f}",
            ]
            rw.writerow(row)
            sw.writerow(row)

            elapsed = time.time() - t0
            print(
                f"  [{idx}/{n_total}] b_BB={vals['b_BB']:.2f} eta={vals['eta']:.2f} "
                f"b_shell={vals['b_shell']:.2f} b_grow={vals['b_grow']:.2f} "
                f"lam={vals['lambda_alpha']:.2f} → "
                f"rec={n_rec} mat={n_mat} mis={n_mis} dup={n_dup} FP={n_fp} "
                f"ce={ce_mean:.3f} re={re_mean:.3f} ve={ve_mean:.3f} "
                f"({elapsed:.1f}s)"
            )

    print(f"Results saved to {results_path}")
    print(f"Summary saved  to {summary_path}")


def _write_error_row(rw, sw, vals, keys):
    row = [vals[k] for k in keys] + [""] * (13 - len(keys))
    rw.writerow(row)
    sw.writerow(row)


if __name__ == "__main__":
    main()
