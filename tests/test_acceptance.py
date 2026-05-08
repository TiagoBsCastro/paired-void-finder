"""Acceptance tests matching the criteria specified in AGENTS.md.

Each test corresponds to one criterion from the ## Acceptance tests section:
1. One-sphere geometry mock recovers one void with center error < 0.2 R_true.
2. Multi-sphere geometry mock recovers approximately the correct number of voids.
3. Close-pair veto mock shows less over-merging with veto enabled than disabled.
4. Periodic-boundary mock recovers one void across the boundary, not two.

(Periodic-distance correctness is covered by test_periodic.py.)

How the geometry mock drives the pipeline
-----------------------------------------
Even with B points strictly inside the sphere and A points outside, the veto
mechanism still fires: B points near the sphere surface have candidate B--B
links whose segments pass close to A points just outside the sphere (within
their local veto radius).  Those links are rejected, fragmenting the surface
B points into singletons while the interior B points remain one connected
component.  The rejected inter-component edges supply the A-boundary shell
needed for the alpha shape.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.validation import validate_against_mock
from paired_void_finder.voids import run_void_finder

# Baseline parameters shared by all pipeline tests (matches algorithm_default.yaml
# except lambda_alpha which is overridden where the sparse geometry mock needs it).
_DEFAULTS = FinderParameters()


def test_one_sphere_center_error_below_threshold():
    """Geometry mock: one void recovered with center error < 0.2 R_true,
    radius within ±50 %, and volume within a factor of 4."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=10000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=1234,
    )
    # lambda_alpha=3.0 is needed here because the sparse B sampling (n~150 inside
    # the sphere) drives a large mean_B_spacing and therefore a large l_BB that
    # fragments the sphere into many small components; a larger R_alpha ensures
    # the alpha shape fills the sphere interior rather than just a thin shell.
    # boundary_mode="shell" is explicit: use the A shell around B members to form
    # the boundary, giving a clean sphere surface without relying on veto records.
    params = replace(_DEFAULTS, lambda_alpha=3.0, boundary_mode="shell")
    voids = run_void_finder(mock.A, mock.B, params)
    assert len(voids) >= 1, f"Expected at least 1 void, got {len(voids)}"
    summary = validate_against_mock(voids, mock)
    assert summary.center_errors.size >= 1, "No recovered void matched the true sphere"
    assert summary.center_errors[0] < 0.2, (
        f"Center error {summary.center_errors[0]:.3f} >= 0.2 R_true"
    )
    assert abs(summary.radius_errors[0]) < 0.1, (
        f"Radius error {summary.radius_errors[0]:.4f} outside ±10%"
    )
    assert abs(summary.volume_errors[0]) < 0.3, (
        f"Volume error {summary.volume_errors[0]:.4f} outside ±30%"
    )


def test_multi_sphere_recovers_approximately_correct_count():
    """Geometry mock: three well-separated spheres yield approximately 3 voids."""
    centers = np.array([[25.0, 25.0, 50.0], [75.0, 50.0, 25.0], [50.0, 75.0, 75.0]])
    radii = np.array([10.0, 10.0, 10.0])
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=15000,
        void_centers=centers,
        void_radii=radii,
        mode="geometry",
        seed=1234,
    )
    voids = run_void_finder(mock.A, mock.B, _DEFAULTS)
    assert 1 <= len(voids) <= 5, (
        f"Expected approximately 3 voids for 3 true spheres, got {len(voids)}"
    )


def test_veto_reduces_over_merging():
    """Veto mock: reduces over-merging (topology) and recovers true spheres (geometry).

    Part 1 — topological check with a CLOSE pair (gap < l_BB):
    Without veto, A points in the inter-sphere gap do not intercept B--B links,
    so interior B points across both spheres can stay linked in one large
    component.  With veto, those cross-gap links are intercepted, fragmenting
    sphere surfaces into singletons and increasing the component count.

    Part 2 — recovery check with a WELL-SEPARATED pair (gap >> l_BB):
    Each sphere has enough B members and A boundary points to be recovered as a
    distinct void.  At least one of the two true spheres must be matched with no
    false positives.
    """
    # ── Part 1: Component count with close pair ──
    close_centers = np.array([[37.0, 50.0, 50.0], [63.0, 50.0, 50.0]])
    close_mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=20000,
        void_centers=close_centers,
        void_radii=np.array([10.0, 10.0]),
        mode="geometry",
        seed=1234,
    )
    _, run_no_veto = run_void_finder(  # type: ignore[misc]
        close_mock.A, close_mock.B, replace(_DEFAULTS, enable_veto=False), return_diagnostics=True
    )
    _, run_veto = run_void_finder(  # type: ignore[misc]
        close_mock.A, close_mock.B, replace(_DEFAULTS, enable_veto=True), return_diagnostics=True
    )
    n_comps_no_veto = len(np.unique(run_no_veto.component_labels))
    n_comps_veto = len(np.unique(run_veto.component_labels))
    assert n_comps_veto > n_comps_no_veto, (
        f"Expected veto to fragment B into more components (less over-merging): "
        f"veto={n_comps_veto}, no_veto={n_comps_no_veto}"
    )

    # ── Part 2: Recovery check with well-separated pair ──
    # Gap (~26) > l_BB (~22) so the FOF cannot merge the two spheres even without
    # veto; reliable recovery is expected with boundary_mode="shell", lambda_alpha=3.0.
    sep_centers = np.array([[25.0, 50.0, 50.0], [75.0, 50.0, 50.0]])
    sep_mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=20000,
        void_centers=sep_centers,
        void_radii=np.array([12.0, 12.0]),
        mode="geometry",
        seed=1234,
    )
    params_recovery = replace(_DEFAULTS, enable_veto=True, boundary_mode="shell", lambda_alpha=3.0)
    voids_sep = run_void_finder(sep_mock.A, sep_mock.B, params_recovery)
    summary = validate_against_mock(voids_sep, sep_mock)
    assert summary.n_matched >= 1, (
        f"Expected at least 1 matched void, got n_matched={summary.n_matched}, "
        f"n_missed={summary.n_missed} from {len(voids_sep)} recovered voids"
    )
    assert summary.false_positive_count == 0, (
        f"Expected no false positives, got {summary.false_positive_count}"
    )


def test_periodic_boundary_void_recovered_as_one():
    """Periodic BC: a void whose sphere straddles the x=0/100 face is one void, not two."""
    # Sphere center at x=2, radius=12 → B points appear at x in [0,14] and x in [90,100].
    # Correct periodic FOF links these into one component; broken PBC would split them.
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=20000,
        void_centers=np.array([[2.0, 50.0, 50.0]]),
        void_radii=np.array([12.0]),
        mode="geometry",
        seed=1234,
    )
    voids = run_void_finder(mock.A, mock.B, _DEFAULTS)
    assert len(voids) == 1, (
        f"Expected exactly 1 void across the periodic boundary, got {len(voids)}"
    )
