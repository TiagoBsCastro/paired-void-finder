"""B--B graph construction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalogs import Edge


def mean_spacing(n_points: int, box_size: float) -> float:
    """Mean inter-point spacing in a cubic periodic box."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    return float((box_size**3 / n_points) ** (1.0 / 3.0))


def candidate_bb_edges(B_positions: np.ndarray, box_size: float, l_BB: float) -> list[Edge]:
    """Return candidate B--B FOF edges using a periodic cKDTree."""
    tree = cKDTree(np.asarray(B_positions, dtype=float), boxsize=box_size)
    pairs = tree.query_pairs(r=l_BB, output_type="set")
    return [Edge(int(i), int(j)) for i, j in sorted(pairs)]
