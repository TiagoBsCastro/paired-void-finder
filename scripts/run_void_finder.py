#!/usr/bin/env python
"""Run the void finder on an NPZ mock."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.catalogs import Catalog, FinderParameters
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
        n_accepted = sum(1 for e in run.edges if e.accepted)
        n_rejected = sum(1 for e in run.edges if not e.accepted)
        comp_ids = np.unique(run.component_labels)
        boundary_sizes = np.array(
            [len(run.final_boundary_sets.get(int(c), [])) for c in comp_ids], dtype=int
        )
        np.savez(
            args.diagnostics,
            A_orig_indices=run.A_orig_indices,
            B_orig_indices=run.B_orig_indices,
            component_labels=run.component_labels,
            n_edges_accepted=np.array([n_accepted]),
            n_edges_rejected=np.array([n_rejected]),
            boundary_sizes=boundary_sizes,
            veto_radii=run.veto_radii if run.veto_radii is not None else np.array([]),
        )
        print(f"Diagnostics saved to {args.diagnostics}")
        print(f"  edges accepted/rejected: {n_accepted}/{n_rejected}")
        print(f"  components: {len(comp_ids)}, final voids: {len(voids)}")


if __name__ == "__main__":
    main()
