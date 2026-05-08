"""Periodic-boundary geometry utilities.

These functions are intentionally simple and heavily tested. More optimized
versions can be added later if needed.
"""

from __future__ import annotations

import numpy as np


def periodic_delta(x1: np.ndarray, x2: np.ndarray, box_size: float) -> np.ndarray:
    """Return minimum-image displacement x1 - x2."""
    delta = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    return delta - box_size * np.rint(delta / box_size)


def periodic_distance(x1: np.ndarray, x2: np.ndarray, box_size: float) -> np.ndarray:
    """Return minimum-image Euclidean distance."""
    delta = periodic_delta(x1, x2, box_size)
    return np.linalg.norm(delta, axis=-1)


def wrap_points(points: np.ndarray, box_size: float) -> np.ndarray:
    """Wrap points into [0, box_size)."""
    return np.asarray(points, dtype=float) % box_size


def unwrap_points(points: np.ndarray, reference: np.ndarray, box_size: float) -> np.ndarray:
    """Unwrap points around a reference using the minimum image."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return reference + periodic_delta(points, reference, box_size)


def distance_point_to_segment_periodic(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    box_size: float,
) -> float:
    """Distance from a point to a segment with periodic boundaries.

    The segment is unwrapped around seg_a using the minimum image, and the point
    is unwrapped around seg_a as well.
    """
    a = np.asarray(seg_a, dtype=float)
    b = a + periodic_delta(seg_b, seg_a, box_size)
    p = a + periodic_delta(point, seg_a, box_size)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))
