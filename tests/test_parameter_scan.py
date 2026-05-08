"""Tests for the parameter scan scripts.

Smoke-tests that run a tiny 2×2 grid and verify the CSV outputs have the
expected shape and column names.  The mock is kept small so the suite
completes in a few seconds.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.validation import validate_against_mock
from paired_void_finder.voids import run_void_finder


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tiny_mock():
    return make_swiss_cheese_mock(
        box_size=50.0,
        n_points=3000,
        void_centers=np.array([[25.0, 25.0, 25.0]]),
        void_radii=np.array([8.0]),
        mode="geometry",
        seed=0,
    )


# ── Unit tests (no subprocess) ────────────────────────────────────────────────


def test_scan_csv_shape_and_columns(tmp_path):
    """Running a 2×2 b_BB×lambda_alpha grid produces a CSV with correct shape."""
    import itertools

    mock = _tiny_mock()
    base = FinderParameters(boundary_mode="shell", enable_veto=False)

    b_BB_vals = [1.2, 1.8]
    lam_vals  = [2.0, 3.0]
    combos    = list(itertools.product(b_BB_vals, lam_vals))

    _HEADER = [
        "b_BB", "eta", "b_shell", "b_grow", "lambda_alpha",
        "n_recovered", "n_matched", "n_missed", "n_duplicate_matches",
        "false_positive_count",
        "mean_center_error", "mean_radius_error", "mean_volume_error",
    ]

    out_path = tmp_path / "results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for b_bb, lam in combos:
            params = FinderParameters(
                boundary_mode="shell", enable_veto=False,
                b_BB=b_bb, lambda_alpha=lam,
            )
            voids   = run_void_finder(mock.A, mock.B, params)
            summary = validate_against_mock(voids, mock)
            n_mat   = summary.n_matched
            ce      = float(np.mean(summary.center_errors)) if n_mat > 0 else float("nan")
            re      = float(np.mean(summary.radius_errors)) if n_mat > 0 else float("nan")
            ve      = float(np.mean(summary.volume_errors)) if n_mat > 0 else float("nan")
            writer.writerow([
                b_bb, base.eta, base.b_shell, base.b_grow, lam,
                summary.n_recovered, n_mat, summary.n_missed,
                summary.n_duplicate_matches, summary.false_positive_count,
                f"{ce:.4f}", f"{re:.4f}", f"{ve:.4f}",
            ])

    # --- Verify CSV structure ---
    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == len(combos), (
        f"Expected {len(combos)} data rows, got {len(rows)}"
    )
    for col in _HEADER:
        assert col in rows[0], f"Missing column {col!r} in CSV output"

    # All b_BB values should be one of the requested values.
    b_vals_found = {float(r["b_BB"]) for r in rows}
    assert b_vals_found == set(b_BB_vals)


def test_scan_all_metrics_numeric_or_nan(tmp_path):
    """Every metric cell is either a parseable float or empty (error row)."""
    mock = _tiny_mock()
    params = FinderParameters(boundary_mode="shell", enable_veto=False, lambda_alpha=2.5)
    voids   = run_void_finder(mock.A, mock.B, params)
    summary = validate_against_mock(voids, mock)

    metric_cols = [
        "n_recovered", "n_matched", "n_missed", "n_duplicate_matches",
        "false_positive_count",
        "mean_center_error", "mean_radius_error", "mean_volume_error",
    ]
    out_path = tmp_path / "single.csv"
    header = ["b_BB", "eta", "b_shell", "b_grow", "lambda_alpha"] + metric_cols
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        n_mat = summary.n_matched
        ce = float(np.mean(summary.center_errors)) if n_mat > 0 else float("nan")
        re = float(np.mean(summary.radius_errors)) if n_mat > 0 else float("nan")
        ve = float(np.mean(summary.volume_errors)) if n_mat > 0 else float("nan")
        writer.writerow([
            params.b_BB, params.eta, params.b_shell, params.b_grow, params.lambda_alpha,
            summary.n_recovered, n_mat, summary.n_missed,
            summary.n_duplicate_matches, summary.false_positive_count,
            f"{ce:.4f}", f"{re:.4f}", f"{ve:.4f}",
        ])

    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    for col in metric_cols:
        val = rows[0][col]
        if val:
            float(val)  # Must not raise.


# ── Integration tests (subprocess) ────────────────────────────────────────────


def test_run_parameter_scan_script_creates_csv(tmp_path):
    """The run_parameter_scan.py script runs and creates both CSV output files."""
    mock_cfg = {
        "box_size": 50.0,
        "n_points": 2500,
        "mode": "geometry",
        "seed": 5,
        "void_centers": [[25.0, 25.0, 25.0]],
        "void_radii": [8.0],
    }
    mock_cfg_path = tmp_path / "mock.yaml"
    with open(mock_cfg_path, "w") as f:
        yaml.dump(mock_cfg, f)

    finder_cfg = _REPO_ROOT / "configs" / "algorithm_default.yaml"
    outdir = tmp_path / "scan_out"
    outdir.mkdir()

    script = _REPO_ROOT / "scripts" / "run_parameter_scan.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--mock-config",   str(mock_cfg_path),
            "--finder-config", str(finder_cfg),
            "--outdir",        str(outdir),
            "--b-BB",          "1.2", "1.8",
            "--lambda-alpha",  "2.0", "3.0",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"Script exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    results_csv = outdir / "param_scan_results.csv"
    summary_csv = outdir / "param_scan_summary.csv"
    assert results_csv.exists(), "param_scan_results.csv not created"
    assert summary_csv.exists(), "param_scan_summary.csv not created"

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 2 b_BB × 2 lambda_alpha = 4 combinations.
    assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"
    assert "b_BB" in rows[0] and "n_matched" in rows[0]


def test_plot_parameter_scan_script_creates_plots(tmp_path):
    """The plot_parameter_scan.py script produces at least line_b_BB.png."""
    import os

    # Write a minimal two-column scan CSV by hand.
    results_path = tmp_path / "param_scan_results.csv"
    header = [
        "b_BB", "eta", "b_shell", "b_grow", "lambda_alpha",
        "n_recovered", "n_matched", "n_missed", "n_duplicate_matches",
        "false_positive_count",
        "mean_center_error", "mean_radius_error", "mean_volume_error",
    ]
    rows = []
    for b_bb in [1.0, 1.5, 2.0]:
        rows.append([b_bb, 0.5, 1.5, 0.5, 2.0, 1, 1, 0, 0, 0, "0.10", "0.05", "0.20"])

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    script = _REPO_ROOT / "scripts" / "plot_parameter_scan.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--results", str(results_path),
            "--outdir",  str(plot_dir),
        ],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, (
        f"Script exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (plot_dir / "line_b_BB.png").exists(), "line_b_BB.png not created"
