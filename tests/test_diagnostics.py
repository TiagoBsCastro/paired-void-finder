"""Tests for diagnostics.py – diagnostic helpers and plots."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.diagnostics import (
    external_faces_from_tetrahedra,
    match_voids_to_truth,
    plot_alpha_diagnostics,
    plot_boundary_size_distribution,
    plot_component_size_distribution,
    plot_radial_profile,
    plot_slice_truth_vs_found,
    plot_xy_projection,
    radial_profile,
    select_best_match,
)
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.voids import run_void_finder

_REPO_ROOT = Path(__file__).parent.parent


# ── external_faces_from_tetrahedra ────────────────────────────────────────────


def test_external_faces_single_tetrahedron():
    """A single tetrahedron has exactly 4 external faces, each with 3 vertices."""
    tets = np.array([[0, 1, 2, 3]])
    faces = external_faces_from_tetrahedra(tets)
    assert faces.shape == (4, 3), f"Expected (4, 3), got {faces.shape}"
    # Each vertex index must be in {0,1,2,3}.
    assert set(faces.ravel().tolist()) == {0, 1, 2, 3}
    # All faces should be unique (sorted rows are distinct).
    unique_rows = np.unique(faces, axis=0)
    assert len(unique_rows) == 4


def test_external_faces_two_tetrahedra_share_face():
    """Two tetrahedra sharing one triangular face produce 6 external faces."""
    # tet1 = [0,1,2,3], tet2 = [0,1,2,4] — they share face (0,1,2).
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    faces = external_faces_from_tetrahedra(tets)
    assert faces.shape[1] == 3
    # Shared face (0,1,2) must not appear in the result.
    shared = np.array([0, 1, 2])
    for f in faces:
        assert not np.array_equal(f, shared), "Shared face (0,1,2) found in external faces"
    assert len(faces) == 6, f"Expected 6 external faces, got {len(faces)}"


# ── radial_profile ────────────────────────────────────────────────────────────


def test_radial_profile_swiss_cheese_inner_density_low():
    """Inside a void sphere, A density should be near zero; B density should be high."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=8000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=42,
    )
    center = mock.true_void_centers[0]
    radius = mock.true_void_radii[0]
    bs = mock.A.box_size

    r_bins = np.linspace(0.0, 2.0 * radius, 20)
    rho_A = radial_profile(mock.A.positions, center, bs, r_bins)
    rho_B = radial_profile(mock.B.positions, center, bs, r_bins)

    # Inside the sphere (r < R_true): A points are absent — the void is empty of A.
    inner_bins = r_bins[1:] < radius
    assert rho_A[inner_bins].max() == 0.0, (
        "Expected zero A density inside the void sphere"
    )

    # Inside the sphere: B density should be positive (B traces the interior).
    assert rho_B[inner_bins].sum() > 0.0, (
        "Expected positive B density inside the void sphere"
    )

    # Outside the sphere (r > R_true): A density should be positive.
    outer_bins = r_bins[:-1] > radius
    assert rho_A[outer_bins].mean() > 0.0, (
        "Expected positive A density outside the void sphere"
    )


# ── Diagnostic plots (smoke tests — check file creation only) ─────────────────


def _small_mock_and_run():
    """Create a small geometry mock and run the finder for smoke tests."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=8000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=42,
    )
    params = FinderParameters(boundary_mode="shell", enable_veto=False, lambda_alpha=3.0)
    voids, run = run_void_finder(mock.A, mock.B, params, return_diagnostics=True)  # type: ignore[misc]
    from paired_void_finder.validation import validate_against_mock
    summary = validate_against_mock(voids, mock)
    return mock, voids, run, summary


# Cache the result to avoid re-running the pipeline for each plot test.
_SMOKE_DATA: tuple | None = None


def _get_smoke_data():
    global _SMOKE_DATA
    if _SMOKE_DATA is None:
        _SMOKE_DATA = _small_mock_and_run()
    return _SMOKE_DATA


def test_diagnostic_script_creates_outputs(tmp_path):
    """Integration test: the script runs and creates all expected output files."""
    mock_cfg = {
        "box_size": 100.0,
        "n_points": 5000,
        "mode": "geometry",
        "seed": 42,
        "void_centers": [[50.0, 50.0, 50.0]],
        "void_radii": [15.0],
    }
    mock_cfg_path = tmp_path / "mock.yaml"
    with open(mock_cfg_path, "w") as f:
        yaml.dump(mock_cfg, f)

    finder_cfg = _REPO_ROOT / "configs" / "algorithm_default.yaml"
    outdir = tmp_path / "out"
    outdir.mkdir()

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    script = _REPO_ROOT / "scripts" / "run_swiss_cheese_diagnostics.py"
    result = __import__("subprocess").run(
        [sys.executable, str(script),
         "--mock-config", str(mock_cfg_path),
         "--finder-config", str(finder_cfg),
         "--outdir", str(outdir)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, (
        f"Script exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    expected_files = [
        "void_catalog.npz",
        "run_diagnostics.npz",
        "summary.txt",
        "match_table.csv",
        "xy_projection.png",
        "slice_z.png",
        "3d_truth_recovered.png",
        "radial_profile.png",
        "component_size_dist.png",
        "boundary_size_dist.png",
        "alpha_diagnostics.png",
    ]
    for name in expected_files:
        assert (outdir / name).exists(), f"Missing expected output file: {name}"

    # Check summary.txt is parseable and contains required keys.
    text = (outdir / "summary.txt").read_text()
    assert "n_true:" in text
    assert "n_recovered:" in text
    assert "n_matched:" in text

    # Check match_table.csv has a header row.
    csv_text = (outdir / "match_table.csv").read_text()
    header = csv_text.splitlines()[0]
    assert "true_id" in header and "void_id" in header


def test_alpha_scan_finds_reasonable_lambda():
    """For a geometry mock, some lambda_alpha in [1.5..5.0] yields n_matched >= 1."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=8000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=42,
    )
    base = FinderParameters(boundary_mode="shell", enable_veto=False)

    found_any = False
    for lam in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        params = replace(base, lambda_alpha=lam)
        voids = run_void_finder(mock.A, mock.B, params)
        summary = match_voids_to_truth(voids, mock)
        if summary.n_matched >= 1:
            found_any = True
            break

    assert found_any, (
        "No lambda_alpha in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] produced n_matched >= 1. "
        "The mock may be too sparse or the alpha shape parameters are misconfigured."
    )
