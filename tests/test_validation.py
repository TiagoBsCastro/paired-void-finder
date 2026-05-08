"""Tests for validate_against_mock: periodic matching, close-pair, one-to-one, duplicates."""

from __future__ import annotations

import numpy as np
import pytest

from paired_void_finder.catalogs import FinderParameters, Void
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.validation import validate_against_mock
from paired_void_finder.voids import run_void_finder

_DEFAULTS = FinderParameters()


def _make_void(void_id: int, center: list[float], radius: float) -> Void:
    """Create a minimal synthetic Void at a known center."""
    c = np.array(center, dtype=float)
    vol = 4.0 * np.pi * radius**3 / 3.0
    return Void(
        void_id=void_id,
        center=c,
        volume=vol,
        effective_radius=radius,
        B_indices=np.array([], dtype=int),
        A_boundary_indices=np.array([], dtype=int),
    )


def test_validate_uses_periodic_distance():
    """validate_against_mock matches a recovered void whose wrapped center
    appears far away in Euclidean space but is close across the periodic boundary.

    True sphere: center [2, 50, 50], R=10.
    Recovered void: center [98, 50, 50] (wrapped equivalent of [-2, 50, 50]).
    Non-periodic distance: 96  >> R  → would be a false positive.
    Periodic distance:       4  < R  → correctly matched.
    """
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=1000,
        void_centers=np.array([[2.0, 50.0, 50.0]]),
        void_radii=np.array([10.0]),
        mode="geometry",
        seed=0,
    )
    # Place the recovered void at x=98 — the "other side" of the boundary.
    synthetic_void = _make_void(0, [98.0, 50.0, 50.0], radius=10.0)
    summary = validate_against_mock([synthetic_void], mock)

    # With periodic distance: d([98,50,50], [2,50,50]) = 4 < 10 → matched.
    assert summary.n_matched == 1
    assert summary.n_missed == 0
    assert summary.n_duplicate_matches == 0
    assert summary.false_positive_count == 0
    assert summary.center_errors.size == 1
    assert summary.center_errors[0] == pytest.approx(4.0 / 10.0, rel=1e-6)
    assert summary.matched_true_indices[0] == 0
    assert summary.matched_void_indices[0] == 0


def test_validate_close_pair_both_found():
    """Two true spheres, two recovered voids, each matched to a different sphere."""
    centers = np.array([[25.0, 50.0, 50.0], [75.0, 50.0, 50.0]])
    radii = np.array([10.0, 10.0])
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=1000,
        void_centers=centers,
        void_radii=radii,
        mode="geometry",
        seed=0,
    )
    # Synthetic voids placed exactly at the true centers.
    void0 = _make_void(0, [25.0, 50.0, 50.0], radius=10.0)
    void1 = _make_void(1, [75.0, 50.0, 50.0], radius=10.0)
    summary = validate_against_mock([void0, void1], mock)

    assert summary.n_matched == 2
    assert summary.n_missed == 0
    assert summary.n_duplicate_matches == 0
    assert summary.false_positive_count == 0
    assert summary.center_errors.size == 2
    assert np.all(summary.center_errors == pytest.approx(0.0, abs=1e-10))
    # Each void matches a distinct true sphere.
    assert len(set(summary.matched_true_indices.tolist())) == 2


def test_validate_one_to_one_matching():
    """Three true spheres: one-to-one LSA assigns each recovered void to a distinct sphere."""
    centers = np.array([[20.0, 20.0, 50.0], [80.0, 50.0, 20.0], [50.0, 80.0, 80.0]])
    radii = np.array([10.0, 10.0, 10.0])
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=1000,
        void_centers=centers,
        void_radii=radii,
        mode="geometry",
        seed=0,
    )

    # Place one void at each true center.
    voids = [_make_void(i, c.tolist(), r) for i, (c, r) in enumerate(zip(centers, radii))]
    summary = validate_against_mock(voids, mock)

    assert summary.n_matched == 3
    assert summary.n_missed == 0
    assert summary.n_duplicate_matches == 0
    assert summary.false_positive_count == 0
    assert summary.center_errors.size == 3
    assert np.all(summary.center_errors == pytest.approx(0.0, abs=1e-10))
    # All three true spheres are covered by distinct recovered voids.
    assert sorted(summary.matched_true_indices.tolist()) == [0, 1, 2]


def test_validate_duplicate_detection():
    """One true sphere, two recovered voids both inside it: one matched, one duplicate.

    The Hungarian algorithm assigns the closer void (void0 at exact center) to
    the true sphere.  void1 at d=2 < R is unmatched by the one-to-one assignment
    but is inside the sphere, so it is counted as a duplicate match, not a
    false positive.
    """
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=1000,
        void_centers=np.array([[50.0, 50.0, 50.0]]),
        void_radii=np.array([10.0]),
        mode="geometry",
        seed=0,
    )
    void0 = _make_void(0, [50.0, 50.0, 50.0], radius=10.0)  # exact match, d=0
    void1 = _make_void(1, [52.0, 50.0, 50.0], radius=10.0)  # d=2 < R, should be duplicate
    summary = validate_against_mock([void0, void1], mock)

    assert summary.n_matched == 1
    assert summary.n_missed == 0
    assert summary.n_duplicate_matches == 1
    assert summary.false_positive_count == 0
    # The better match (void0) should be the one assigned.
    assert summary.matched_void_indices[0] == 0
    assert summary.matched_true_indices[0] == 0


def test_validate_periodic_boundary_end_to_end():
    """End-to-end: a void straddling x=0/100 is recovered and not a false positive."""
    mock = make_swiss_cheese_mock(
        box_size=100.0,
        n_points=20000,
        void_centers=np.array([[2.0, 50.0, 50.0]]),
        void_radii=np.array([12.0]),
        mode="geometry",
        seed=1234,
    )
    voids = run_void_finder(mock.A, mock.B, _DEFAULTS)
    assert len(voids) >= 1, f"Expected at least 1 void, got {len(voids)}"
    summary = validate_against_mock(voids, mock)
    # The void must be matched (not classified as a false positive).
    assert summary.false_positive_count == 0, (
        f"Void near periodic boundary was a false positive — "
        f"validate_against_mock may not be using periodic distance"
    )
    assert summary.center_errors.size >= 1
