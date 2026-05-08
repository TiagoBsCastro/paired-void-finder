"""Validation metrics for recovered voids against mock truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .catalogs import MockCatalog, Void
from .periodic import periodic_distance


@dataclass(slots=True)
class ValidationSummary:
    n_true: int
    n_recovered: int
    n_matched: int                   # # true spheres covered by a unique matched void
    n_missed: int                    # # true spheres with no matched void
    n_duplicate_matches: int         # # recovered voids inside a sphere already matched
    matched_true_indices: np.ndarray # indices of matched true spheres (sorted)
    matched_void_indices: np.ndarray # corresponding recovered-void indices (parallel)
    center_errors: np.ndarray        # d / R_true for each matched pair (parallel to above)
    radius_errors: np.ndarray        # (R_rec - R_true) / R_true for each matched pair
    volume_errors: np.ndarray        # (V_rec - V_true) / V_true for each matched pair
    false_positive_count: int        # recovered voids outside all true spheres
    largest_volume_fraction: float


def validate_against_mock(voids: list[Void], mock: MockCatalog) -> ValidationSummary:
    """One-to-one Hungarian matching of recovered voids against known spherical voids.

    The cost matrix entry C[i, j] is the normalised periodic distance
    ``d(recovered_i, true_j) / R_true_j``.  Entries where the recovered void
    centre lies outside the true sphere (d > R_true_j) are forbidden by setting
    them to a large sentinel.  ``scipy.optimize.linear_sum_assignment`` then
    finds the minimum-cost one-to-one assignment.

    After the assignment:
    - Valid matched pairs have C[i, j] < sentinel / 2.
    - Unmatched recovered voids inside any true sphere are counted as duplicates.
    - Unmatched recovered voids outside all true spheres are false positives.
    """
    n_true = len(mock.true_void_radii)
    n_rec = len(voids)
    box_size = mock.A.box_size

    _empty = ValidationSummary(
        n_true=n_true,
        n_recovered=0,
        n_matched=0,
        n_missed=n_true,
        n_duplicate_matches=0,
        matched_true_indices=np.array([], dtype=int),
        matched_void_indices=np.array([], dtype=int),
        center_errors=np.array([]),
        radius_errors=np.array([]),
        volume_errors=np.array([]),
        false_positive_count=0,
        largest_volume_fraction=0.0,
    )
    if n_rec == 0 or n_true == 0:
        if n_rec > 0:
            # All recovered voids are false positives when there are no true spheres.
            rec_volumes = np.asarray([v.volume for v in voids])
            return ValidationSummary(
                n_true=n_true,
                n_recovered=n_rec,
                n_matched=0,
                n_missed=n_true,
                n_duplicate_matches=0,
                matched_true_indices=np.array([], dtype=int),
                matched_void_indices=np.array([], dtype=int),
                center_errors=np.array([]),
                radius_errors=np.array([]),
                volume_errors=np.array([]),
                false_positive_count=n_rec,
                largest_volume_fraction=float(np.max(rec_volumes) / box_size**3),
            )
        return _empty

    rec_centers = np.asarray([v.center for v in voids])
    rec_radii = np.asarray([v.effective_radius for v in voids])
    rec_volumes = np.asarray([v.volume for v in voids])

    _FORBIDDEN = 1e10
    cost = np.full((n_rec, n_true), _FORBIDDEN)
    for i, center in enumerate(rec_centers):
        d = periodic_distance(mock.true_void_centers, center, box_size)
        for j in range(n_true):
            if d[j] <= mock.true_void_radii[j]:
                cost[i, j] = d[j] / mock.true_void_radii[j]

    row_ind, col_ind = linear_sum_assignment(cost)

    # Extract valid pairs and sort by true-sphere index for deterministic output.
    pairs: list[tuple[int, int, float]] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < _FORBIDDEN / 2:
            pairs.append((int(i), int(j), float(cost[i, j])))
    pairs.sort(key=lambda x: x[1])

    matched_void_indices = np.array([p[0] for p in pairs], dtype=int)
    matched_true_indices = np.array([p[1] for p in pairs], dtype=int)
    center_errors = np.array([p[2] for p in pairs])
    radius_errors = np.array(
        [(rec_radii[p[0]] - mock.true_void_radii[p[1]]) / mock.true_void_radii[p[1]]
         for p in pairs]
    )
    volume_errors = np.array(
        [(rec_volumes[p[0]] - mock.true_void_volumes[p[1]]) / mock.true_void_volumes[p[1]]
         for p in pairs]
    )

    matched_void_set = set(matched_void_indices.tolist())
    n_matched = len(pairs)
    n_missed = n_true - len(set(matched_true_indices.tolist()))

    n_duplicate = 0
    n_fp = 0
    for i in range(n_rec):
        if i in matched_void_set:
            continue
        d = periodic_distance(mock.true_void_centers, rec_centers[i], box_size)
        if np.any(d <= mock.true_void_radii):
            n_duplicate += 1
        else:
            n_fp += 1

    return ValidationSummary(
        n_true=n_true,
        n_recovered=n_rec,
        n_matched=n_matched,
        n_missed=n_missed,
        n_duplicate_matches=n_duplicate,
        matched_true_indices=matched_true_indices,
        matched_void_indices=matched_void_indices,
        center_errors=center_errors,
        radius_errors=radius_errors,
        volume_errors=volume_errors,
        false_positive_count=n_fp,
        largest_volume_fraction=float(np.max(rec_volumes) / box_size**3),
    )
