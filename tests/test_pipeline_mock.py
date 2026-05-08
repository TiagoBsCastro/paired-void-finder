from dataclasses import replace

import numpy as np

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.validation import validate_against_mock
from paired_void_finder.voids import run_void_finder


def test_pipeline_recovers_one_known_sphere():
    """Pipeline correctly recovers one known sphere and returns diagnostics."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=10000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=42,
    )
    params = FinderParameters(lambda_alpha=3.0)
    voids, run = run_void_finder(mock.A, mock.B, params, return_diagnostics=True)

    # Structural checks on the diagnostic record.
    assert run.A_orig_indices.dtype == int
    assert run.B_orig_indices.dtype == int
    assert len(run.component_labels) == len(run.B_orig_indices)
    assert isinstance(run.edges, list)
    assert run.veto_radii is not None  # veto was enabled by default

    # At least one void should be recovered.
    assert len(voids) >= 1, f"Expected at least 1 void, got {len(voids)}"

    # The recovered void's B_indices and A_boundary_indices should be original
    # catalog indices, not sub-array indices.
    v = voids[0]
    assert np.all(v.B_indices < len(mock.B.positions)), "B_indices out of range"
    assert np.all(v.A_boundary_indices < len(mock.A.positions)), "A_boundary_indices out of range"

    # Validate center error.
    summary = validate_against_mock(voids, mock)
    assert summary.center_errors.size >= 1
    assert summary.center_errors[0] < 0.3, (
        f"Center error {summary.center_errors[0]:.3f} R_true >= 0.3"
    )


def test_pipeline_veto_mode_decoys_not_in_a():
    """Veto-mode decoy B points must not coincide with any A point."""
    mock = make_swiss_cheese_mock(
        box_size=50.0,
        n_points=4000,
        void_centers=np.array([[25.0, 25.0, 25.0]]),
        void_radii=np.array([8.0]),
        mode="veto",
        exterior_b_fraction=0.05,
        seed=7,
    )
    A_set = set(map(tuple, np.round(mock.A.positions, 10)))
    B_set = set(map(tuple, np.round(mock.B.positions, 10)))
    overlap = A_set & B_set
    assert len(overlap) == 0, f"Found {len(overlap)} positions shared between A and B"


def test_pipeline_original_indices_preserved_after_mass_cut():
    """Void.B_indices and Void.A_boundary_indices index the original catalog."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=10000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([15.0]),
        mode="geometry",
        seed=42,
    )
    # Add masses so that a mass cut removes half the points.
    n_A = len(mock.A.positions)
    n_B = len(mock.B.positions)
    masses_A = np.where(np.arange(n_A) % 2 == 0, 1.0, 0.5)
    masses_B = np.where(np.arange(n_B) % 2 == 0, 1.0, 0.5)
    from paired_void_finder.catalogs import Catalog
    A_cut = Catalog(mock.A.positions, masses_A, mock.A.box_size, name="A")
    B_cut = Catalog(mock.B.positions, masses_B, mock.B.box_size, name="B")
    params = FinderParameters(M_A_min=0.9, M_B_min=0.9, lambda_alpha=3.0)
    voids = run_void_finder(A_cut, B_cut, params)
    for v in voids:
        assert np.all(v.B_indices < n_B), "B_indices exceed original B size"
        assert np.all(v.A_boundary_indices < n_A), "A_boundary_indices exceed original A size"
        # All referenced B positions should have passed the mass cut.
        assert np.all(masses_B[v.B_indices] >= 0.9)
        assert np.all(masses_A[v.A_boundary_indices] >= 0.9)
