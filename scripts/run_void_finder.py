#!/usr/bin/env python
"""Run the void finder on an NPZ mock."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.catalogs import Catalog, FinderParameters, FinderRun
from paired_void_finder.voids import run_void_finder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True)
    parser.add_argument("--config", default="configs/algorithm_default.yaml")
    parser.add_argument("--out", default="voids.npz")
    parser.add_argument(
        "--diagnostics",
        default=None,
        metavar="FILE",
        help="If provided, save intermediate diagnostic products to this NPZ file.",
    )
    args = parser.parse_args()

    data = np.load(args.mock)
    box_size = float(data["box_size"])
    A = Catalog(data["A_positions"], data["A_masses"], box_size, name="A")
    B = Catalog(data["B_positions"], data["B_masses"], box_size, name="B")
    params = FinderParameters.from_yaml(args.config)

    return_diag = args.diagnostics is not None
    result = run_void_finder(A, B, params, return_diagnostics=return_diag)
    if return_diag:
        voids, run = result  # type: ignore[misc]
    else:
        voids = result  # type: ignore[assignment]

    np.savez(
        args.out,
        centers=np.asarray([v.center for v in voids]) if voids else np.empty((0, 3)),
        volumes=np.asarray([v.volume for v in voids]),
        effective_radii=np.asarray([v.effective_radius for v in voids]),
    )
    print(f"Recovered {len(voids)} voids")
    print(f"Saved {args.out}")

    if return_diag:
        _save_diagnostics(args.diagnostics, run, voids)
        print(f"Diagnostics saved to {args.diagnostics}")


def _ragged_boundary(
    d: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialise a {comp_id: A_indices} dict to three flat arrays."""
    if not d:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    comp_ids = np.array(sorted(d.keys()), dtype=int)
    counts = np.array([len(d[c]) for c in comp_ids], dtype=int)
    flat = np.concatenate([d[c] for c in comp_ids]) if counts.sum() > 0 else np.array([], dtype=int)
    return comp_ids, counts, flat


def _save_diagnostics(path: str, run: FinderRun, voids: list) -> None:
    """Save full edge and boundary diagnostic arrays to an NPZ file."""
    # --- edges ---
    edges = run.edges
    if edges:
        edge_i = np.array([e.i for e in edges], dtype=int)
        edge_j = np.array([e.j for e in edges], dtype=int)
        edge_accepted = np.array([e.accepted for e in edges], dtype=bool)
        halo_counts = np.array([len(e.veto_halos) for e in edges], dtype=int)
        halos_flat = (
            np.concatenate([np.asarray(e.veto_halos, dtype=int) for e in edges
                            if len(e.veto_halos) > 0])
            if any(len(e.veto_halos) > 0 for e in edges)
            else np.array([], dtype=int)
        )
    else:
        edge_i = edge_j = np.array([], dtype=int)
        edge_accepted = np.array([], dtype=bool)
        halo_counts = halos_flat = np.array([], dtype=int)

    # --- boundary sets ---
    veto_ids, veto_counts, veto_flat = _ragged_boundary(run.veto_boundary_sets)
    shell_ids, shell_counts, shell_flat = _ragged_boundary(run.shell_boundary_sets)
    sel_ids, sel_counts, sel_flat = _ragged_boundary(run.selected_boundary_sets)
    fin_ids, fin_counts, fin_flat = _ragged_boundary(run.final_boundary_sets)

    n_accepted = int(edge_accepted.sum()) if len(edge_accepted) else 0
    n_rejected = int((~edge_accepted).sum()) if len(edge_accepted) else 0
    comp_ids = np.unique(run.component_labels)

    np.savez(
        path,
        # index maps
        A_orig_indices=run.A_orig_indices,
        B_orig_indices=run.B_orig_indices,
        component_labels=run.component_labels,
        veto_radii=run.veto_radii if run.veto_radii is not None else np.array([]),
        # edge arrays
        edge_i=edge_i,
        edge_j=edge_j,
        edge_accepted=edge_accepted,
        edge_veto_halo_counts=halo_counts,
        edge_veto_halos_flat=halos_flat,
        # summary scalars
        n_edges_accepted=np.array([n_accepted]),
        n_edges_rejected=np.array([n_rejected]),
        n_components=np.array([len(comp_ids)]),
        n_voids=np.array([len(voids)]),
        # veto boundary
        veto_boundary_comp_ids=veto_ids,
        veto_boundary_counts=veto_counts,
        veto_boundary_flat=veto_flat,
        # shell boundary
        shell_boundary_comp_ids=shell_ids,
        shell_boundary_counts=shell_counts,
        shell_boundary_flat=shell_flat,
        # selected boundary (before dilation)
        selected_boundary_comp_ids=sel_ids,
        selected_boundary_counts=sel_counts,
        selected_boundary_flat=sel_flat,
        # final boundary (after dilation)
        final_boundary_comp_ids=fin_ids,
        final_boundary_counts=fin_counts,
        final_boundary_flat=fin_flat,
    )
    print(f"  edges accepted/rejected: {n_accepted}/{n_rejected}")
    print(f"  components: {len(comp_ids)}, final voids: {len(voids)}")


if __name__ == "__main__":
    main()
