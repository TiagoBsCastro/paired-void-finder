"""Validation metrics for recovered voids against mock truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalogs import MockCatalog, Void
from .periodic import periodic_distance


@dataclass(slots=True)
class ValidationSummary:
    n_true: int
    n_recovered: int
    center_errors: np.ndarray
    radius_errors: np.ndarray
    volume_errors: np.ndarray
    false_positive_count: int
    largest_volume_fraction: float


def validate_against_mock(voids: list[Void], mock: MockCatalog) -> ValidationSummary:
    """Nearest-center matching validation against known spherical voids."""
    if len(voids) == 0:
        return ValidationSummary(
            n_true=len(mock.true_void_radii),
            n_recovered=0,
            center_errors=np.array([]),
            radius_errors=np.array([]),
            volume_errors=np.array([]),
            false_positive_count=0,
            largest_volume_fraction=0.0,
        )
    centers = np.asarray([v.center for v in voids])
    radii = np.asarray([v.effective_radius for v in voids])
    volumes = np.asarray([v.volume for v in voids])
    matched = []
    false_pos = 0
    center_errors = []
    radius_errors = []
    volume_errors = []
    for center, radius, volume in zip(centers, radii, volumes):
        d = periodic_distance(mock.true_void_centers, center, mock.A.box_size)
        j = int(np.argmin(d))
        if d[j] > mock.true_void_radii[j]:
            false_pos += 1
            continue
        matched.append(j)
        center_errors.append(d[j] / mock.true_void_radii[j])
        radius_errors.append((radius - mock.true_void_radii[j]) / mock.true_void_radii[j])
        volume_errors.append((volume - mock.true_void_volumes[j]) / mock.true_void_volumes[j])
    return ValidationSummary(
        n_true=len(mock.true_void_radii),
        n_recovered=len(voids),
        center_errors=np.asarray(center_errors),
        radius_errors=np.asarray(radius_errors),
        volume_errors=np.asarray(volume_errors),
        false_positive_count=false_pos,
        largest_volume_fraction=float(np.max(volumes) / mock.A.box_size**3),
    )
