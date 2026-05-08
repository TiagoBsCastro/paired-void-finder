"""A-barrier veto logic for candidate B--B edges."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalogs import Edge
from .periodic import distance_point_to_segment_periodic, periodic_delta


def compute_veto_radii(
    A_positions: np.ndarray,
    box_size: float,
    eta: float,
    N_veto: int,
) -> np.ndarray:
    """Compute r_veto,k = eta * d_N,k^A for all A points."""
    if N_veto < 1:
        raise ValueError("N_veto must be >= 1")
    A_positions = np.asarray(A_positions, dtype=float)
    if len(A_positions) <= N_veto:
        raise ValueError("Need more A points than N_veto")
    tree = cKDTree(A_positions, boxsize=box_size)
    # k includes the point itself at distance zero, so request N_veto + 1.
    distances, _ = tree.query(A_positions, k=N_veto + 1)
    d_N = distances[:, -1]
    return eta * d_N


def apply_a_barrier_veto(
    B_positions: np.ndarray,
    A_positions: np.ndarray,
    candidate_edges: list[Edge],
    veto_radii: np.ndarray,
    box_size: float,
) -> list[Edge]:
    """Reject B--B links that pass within r_veto of an A point.

    A periodic KDTree selects only A points that could plausibly veto each
    segment. The exact point-to-segment distance is still evaluated afterward.
    """
    accepted_or_rejected: list[Edge] = []
    A_positions = np.asarray(A_positions, dtype=float)
    B_positions = np.asarray(B_positions, dtype=float)
    veto_radii = np.asarray(veto_radii, dtype=float)
    tree = cKDTree(A_positions, boxsize=box_size)
    max_radius = float(np.max(veto_radii)) if len(veto_radii) else 0.0

    for edge in candidate_edges:
        a = B_positions[edge.i]
        delta = periodic_delta(B_positions[edge.j], a, box_size)
        segment_length = float(np.linalg.norm(delta))
        midpoint = (a + 0.5 * delta) % box_size
        search_radius = 0.5 * segment_length + max_radius
        candidate_a = tree.query_ball_point(midpoint, r=search_radius)

        veto_ids: list[int] = []
        for k in candidate_a:
            d = distance_point_to_segment_periodic(A_positions[k], a, B_positions[edge.j], box_size)
            if d < veto_radii[k]:
                veto_ids.append(int(k))

        accepted_or_rejected.append(
            Edge(edge.i, edge.j, accepted=(len(veto_ids) == 0), veto_halos=tuple(sorted(veto_ids)))
        )
    return accepted_or_rejected
