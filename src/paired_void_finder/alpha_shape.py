"""3D alpha-shape geometry."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from .catalogs import AlphaShapeResult
from .periodic import unwrap_points, wrap_points


def tetra_volume(points: np.ndarray) -> float:
    """Volume of one tetrahedron with shape (4, 3)."""
    p1, p2, p3, p4 = points
    return float(abs(np.linalg.det(np.stack([p2 - p1, p3 - p1, p4 - p1], axis=1))) / 6.0)


def circumsphere_radius_tetra(points: np.ndarray) -> float:
    """Circumsphere radius of one tetrahedron.

    Returns infinity for degenerate tetrahedra.
    """
    p0 = points[0]
    A = 2.0 * (points[1:] - p0)
    b = np.sum(points[1:] ** 2 - p0**2, axis=1)
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.linalg.norm(center - p0))


def alpha_shape_3d(
    points: np.ndarray,
    box_size: float,
    R_alpha: float,
    reference: np.ndarray | None = None,
) -> AlphaShapeResult:
    """Compute a simple 3D alpha shape from a point set.

    The input points are unwrapped around `reference`, Delaunay tetrahedra are
    computed, and tetrahedra with circumsphere radius below `R_alpha` are kept.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError("Need at least four points for a 3D alpha shape")
    if reference is None:
        reference = points[0]
    unwrapped = unwrap_points(points, np.asarray(reference, dtype=float), box_size)
    delaunay = Delaunay(unwrapped)
    tetrahedra = np.asarray(delaunay.simplices, dtype=int)
    keep: list[int] = []
    volumes: list[float] = []
    centroids: list[np.ndarray] = []
    for t_idx, tet in enumerate(tetrahedra):
        tet_points = unwrapped[tet]
        radius = circumsphere_radius_tetra(tet_points)
        if radius < R_alpha:
            vol = tetra_volume(tet_points)
            if vol > 0:
                keep.append(t_idx)
                volumes.append(vol)
                centroids.append(np.mean(tet_points, axis=0))
    n_total = len(tetrahedra)
    n_accepted = len(keep)
    if n_accepted == 0:
        raise ValueError("Alpha shape has no accepted tetrahedra")
    v = np.asarray(volumes)
    c = np.asarray(centroids)
    total_volume = float(np.sum(v))
    centroid_unwrapped = np.sum(c * v[:, None], axis=0) / total_volume
    centroid_wrapped = wrap_points(centroid_unwrapped, box_size)
    R_eff = float((3.0 * total_volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    return AlphaShapeResult(
        vertices=unwrapped,
        tetrahedra=tetrahedra,
        accepted_tetrahedra=tetrahedra[np.asarray(keep, dtype=int)],
        volume=total_volume,
        centroid_unwrapped=centroid_unwrapped,
        centroid_wrapped=centroid_wrapped,
        effective_radius=R_eff,
        n_tetrahedra_total=n_total,
        n_tetrahedra_accepted=n_accepted,
        alpha_fraction=float(n_accepted / n_total) if n_total > 0 else 0.0,
    )
