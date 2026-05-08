"""High-level void-finder pipeline."""

from __future__ import annotations

from typing import overload

import numpy as np

from .alpha_shape import alpha_shape_3d
from .boundaries import (
    component_labels_from_edges,
    dilate_boundaries,
    shell_boundaries_by_component,
    veto_boundaries_by_component,
)
from .catalogs import Catalog, FinderParameters, FinderRun, Void
from .graph import candidate_bb_edges, mean_spacing
from .periodic import unwrap_points
from .veto import apply_a_barrier_veto, compute_veto_radii


def run_void_finder(
    A: Catalog,
    B: Catalog,
    params: FinderParameters,
    return_diagnostics: bool = False,
) -> "list[Void] | tuple[list[Void], FinderRun]":
    """Run the first-pass paired barrier void finder.

    Parameters
    ----------
    A, B:
        Barrier and tracer catalogs.
    params:
        Algorithm parameters.
    return_diagnostics:
        When True, also return a :class:`FinderRun` with intermediate products.
    """
    A_mask = np.ones(len(A.positions), dtype=bool)
    if A.masses is not None:
        A_mask = A.masses >= params.M_A_min
    B_mask = np.ones(len(B.positions), dtype=bool)
    if B.masses is not None:
        B_mask = B.masses >= params.M_B_min

    # Guard: veto boundary mode requires the veto mechanism to be enabled.
    if not params.enable_veto and params.boundary_mode == "veto":
        raise ValueError(
            "boundary_mode='veto' requires enable_veto=True — there are no rejected "
            "edges to build veto boundaries from when the veto is disabled. "
            "Use boundary_mode='shell' or 'hybrid' when enable_veto=False."
        )

    # Preserve mapping from sub-array index → original catalog index.
    A_orig_indices = np.flatnonzero(A_mask)
    B_orig_indices = np.flatnonzero(B_mask)

    A_pos = A.positions[A_mask]
    B_pos = B.positions[B_mask]
    box_size = A.box_size
    dA = mean_spacing(len(A_pos), box_size)
    dB = mean_spacing(len(B_pos), box_size)
    l_BB = params.b_BB * dB

    edges = candidate_bb_edges(B_pos, box_size, l_BB)
    veto_radii: np.ndarray | None = None
    if params.enable_veto:
        veto_radii = compute_veto_radii(A_pos, box_size, params.eta, params.N_veto)
        edges = apply_a_barrier_veto(B_pos, A_pos, edges, veto_radii, box_size)

    labels = component_labels_from_edges(len(B_pos), edges)

    # Build raw (pre-dilation) boundary sets according to boundary_mode.
    mode = params.boundary_mode
    l_shell = params.b_shell * dA
    veto_bs: dict[int, np.ndarray] = {}
    shell_bs: dict[int, np.ndarray] = {}

    if mode in ("veto", "hybrid"):
        veto_bs = veto_boundaries_by_component(edges, labels)
    if mode in ("shell", "hybrid"):
        shell_bs = shell_boundaries_by_component(B_pos, A_pos, labels, box_size, l_shell)

    if mode == "veto":
        selected_bs = veto_bs
    elif mode == "shell":
        selected_bs = shell_bs
    else:  # hybrid: prefer veto when it has enough points, else fall back to shell
        selected_bs = {
            c: veto_bs[c] if len(veto_bs[c]) >= params.N_A_min else shell_bs[c]
            for c in veto_bs
        }

    final_boundaries = selected_bs
    if params.b_grow > 0:
        final_boundaries = dilate_boundaries(A_pos, selected_bs, box_size, params.b_grow * dA)

    voids: list[Void] = []
    R_alpha = params.lambda_alpha * dA
    for comp_id in sorted(np.unique(labels)):
        B_sub_indices = np.flatnonzero(labels == comp_id)
        if len(B_sub_indices) < params.N_B_min:
            continue
        A_boundary_sub = final_boundaries.get(int(comp_id), np.array([], dtype=int))
        if len(A_boundary_sub) < params.N_A_min:
            continue

        # Unwrap B positions around the first member so the mean is correct
        # even when the component straddles a periodic boundary.
        seed = B_pos[B_sub_indices[0]]
        unwrapped_B = unwrap_points(B_pos[B_sub_indices], seed, box_size)
        reference = np.mean(unwrapped_B, axis=0) % box_size

        try:
            shape = alpha_shape_3d(A_pos[A_boundary_sub], box_size, R_alpha, reference=reference)
        except ValueError:
            continue
        if shape.effective_radius < params.R_min:
            continue

        # Map sub-array indices back to original catalog indices.
        B_orig = B_orig_indices[B_sub_indices]
        A_orig = A_orig_indices[A_boundary_sub]

        voids.append(
            Void(
                void_id=len(voids),
                center=shape.centroid_wrapped,
                volume=shape.volume,
                effective_radius=shape.effective_radius,
                B_indices=B_orig,
                A_boundary_indices=A_orig,
                alpha_shape=shape,
                metadata={"component_id": int(comp_id)},
            )
        )

    if not return_diagnostics:
        return voids

    # When returning diagnostics, also compute the boundary set that wasn't needed
    # by the main pipeline path so all four fields are always populated.
    if mode == "veto":
        shell_bs = shell_boundaries_by_component(B_pos, A_pos, labels, box_size, l_shell)
    elif mode == "shell":
        veto_bs = veto_boundaries_by_component(edges, labels)

    run = FinderRun(
        params=params,
        A_orig_indices=A_orig_indices,
        B_orig_indices=B_orig_indices,
        edges=edges,
        veto_radii=veto_radii,
        component_labels=labels,
        veto_boundary_sets=veto_bs,
        shell_boundary_sets=shell_bs,
        selected_boundary_sets=selected_bs,
        final_boundary_sets=final_boundaries,
        voids=voids,
    )
    return voids, run
