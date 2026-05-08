"""Boundary extraction from veto records."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalogs import Edge


def component_labels_from_edges(n_nodes: int, edges: list[Edge]) -> np.ndarray:
    """Connected-component labels for accepted B--B edges."""
    parent = np.arange(n_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        if edge.accepted:
            union(edge.i, edge.j)
    roots = np.array([find(i) for i in range(n_nodes)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def veto_boundaries_by_component(edges: list[Edge], component_labels: np.ndarray) -> dict[int, np.ndarray]:
    """Build A-boundary sets from rejected inter-component edges."""
    out: dict[int, set[int]] = {int(c): set() for c in np.unique(component_labels)}
    for edge in edges:
        if edge.accepted or len(edge.veto_halos) == 0:
            continue
        ci = int(component_labels[edge.i])
        cj = int(component_labels[edge.j])
        if ci == cj:
            continue
        out[ci].update(edge.veto_halos)
        out[cj].update(edge.veto_halos)
    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


def dilate_boundaries(
    A_positions: np.ndarray,
    boundary_sets: dict[int, np.ndarray],
    box_size: float,
    l_grow: float,
) -> dict[int, np.ndarray]:
    """Add A points within l_grow of the current boundary points."""
    if l_grow <= 0:
        return boundary_sets
    tree = cKDTree(A_positions, boxsize=box_size)
    grown: dict[int, np.ndarray] = {}
    for comp_id, boundary in boundary_sets.items():
        if len(boundary) == 0:
            grown[comp_id] = boundary
            continue
        near: set[int] = set(map(int, boundary))
        for idx in boundary:
            near.update(map(int, tree.query_ball_point(A_positions[idx], r=l_grow)))
        grown[comp_id] = np.array(sorted(near), dtype=int)
    return grown
