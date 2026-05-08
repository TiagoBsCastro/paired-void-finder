"""High-level void-finder pipeline."""

from __future__ import annotations

import numpy as np

from .alpha_shape import alpha_shape_3d
from .boundaries import component_labels_from_edges, dilate_boundaries, veto_boundaries_by_component
from .catalogs import Catalog, FinderParameters, Void
from .graph import candidate_bb_edges, mean_spacing
from .veto import apply_a_barrier_veto, compute_veto_radii


def run_void_finder(A: Catalog, B: Catalog, params: FinderParameters) -> list[Void]:
    """Run the first-pass paired barrier void finder.

    This is intentionally simple and intended for mock validation.
    """
    A_mask = np.ones(len(A.positions), dtype=bool)
    if A.masses is not None:
        A_mask = A.masses >= params.M_A_min
    B_mask = np.ones(len(B.positions), dtype=bool)
    if B.masses is not None:
        B_mask = B.masses >= params.M_B_min

    A_pos = A.positions[A_mask]
    B_pos = B.positions[B_mask]
    box_size = A.box_size
    dA = mean_spacing(len(A_pos), box_size)
    dB = mean_spacing(len(B_pos), box_size)
    l_BB = params.b_BB * dB
    edges = candidate_bb_edges(B_pos, box_size, l_BB)
    if params.enable_veto:
        veto_radii = compute_veto_radii(A_pos, box_size, params.eta, params.N_veto)
        edges = apply_a_barrier_veto(B_pos, A_pos, edges, veto_radii, box_size)
    labels = component_labels_from_edges(len(B_pos), edges)
    boundary_sets = veto_boundaries_by_component(edges, labels)
    if params.b_grow > 0:
        boundary_sets = dilate_boundaries(A_pos, boundary_sets, box_size, params.b_grow * dA)

    voids: list[Void] = []
    R_alpha = params.lambda_alpha * dA
    for comp_id in sorted(np.unique(labels)):
        B_indices = np.flatnonzero(labels == comp_id)
        if len(B_indices) < params.N_B_min:
            continue
        A_boundary = boundary_sets.get(int(comp_id), np.array([], dtype=int))
        if len(A_boundary) < params.N_A_min:
            continue
        reference = np.mean(B_pos[B_indices], axis=0) % box_size
        try:
            shape = alpha_shape_3d(A_pos[A_boundary], box_size, R_alpha, reference=reference)
        except ValueError:
            continue
        if shape.effective_radius < params.R_min:
            continue
        voids.append(
            Void(
                void_id=len(voids),
                center=shape.centroid_wrapped,
                volume=shape.volume,
                effective_radius=shape.effective_radius,
                B_indices=B_indices,
                A_boundary_indices=A_boundary,
                alpha_shape=shape,
                metadata={"component_id": int(comp_id)},
            )
        )
    return voids
