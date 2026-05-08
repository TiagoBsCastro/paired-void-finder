# Agent instructions

Implement a Python package called `paired_void_finder`.

## Scientific goal

Build a barrier-aware paired-catalog void finder. Given two catalogs `A` and `B` in a periodic cubic box:

- `B` points trace candidate void interiors in `A`.
- `A` points act as barriers.
- Candidate `B--B` FOF links are vetoed if the segment between the two `B` points passes within a local veto radius of any `A` point.
- The local veto radius is

  `r_veto,k = eta * d_N,k^A`,

  where `d_N,k^A` is the distance from `A_k` to its `N_veto`-th nearest `A` neighbor.
- Store the `A` points responsible for each rejected `B--B` link.
- Connected components of the accepted `B--B` graph are protovoids.
- The `A` boundary of each protovoid is the union of vetoing `A` points on rejected edges connecting that component to another component.
- Optionally dilate the boundary by adding nearby `A` points within `l_grow = b_grow * mean_A_spacing`.
- Compute a 3D alpha shape from the final boundary set.
- Keep Delaunay tetrahedra with circumsphere radius smaller than `R_alpha = lambda_alpha * mean_A_spacing`.
- Compute volume and volume centroid from the accepted tetrahedra.

## Implementation constraints

- Prioritize correctness, readability, and tests over speed.
- All distances must use periodic boundary conditions.
- Use dataclasses for catalogs, parameters, edges, alpha-shape results, and void objects.
- Use NPZ files for initial I/O.
- Save intermediate products where possible: candidate edges, rejected edges, veto halos, component labels, boundary sets, and final void catalog.
- Do not implement Pinocchio reading in the first pass. Add a placeholder interface only.

## Required dependencies

Use only standard Python plus: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `networkx`, and `pytest`.

## Acceptance tests

- Periodic distances are correct for points across the box boundary.
- One-sphere geometry mock recovers one void with center error below `0.2 R_true` after reasonable default parameters.
- Multi-sphere geometry mock recovers approximately the correct number of voids.
- Close-pair veto mock shows less over-merging with veto enabled than with veto disabled.
- Periodic-boundary mock recovers one void across the boundary, not two.
