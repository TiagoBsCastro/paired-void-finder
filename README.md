# Paired Void Finder

Starter repository for a paired-catalog barrier void finder.

The algorithm uses two catalogs, `A` and `B`, in a periodic cubic box. Catalog `B` traces candidate void interiors in `A`. Catalog `A` acts as a barrier. Candidate `B--B` links are vetoed when the segment between two `B` points passes too close to an `A` point. The `A` points responsible for vetoing rejected links define the boundary of each recovered void. The void geometry is built using a 3D alpha shape, and the center is the volume centroid of accepted Delaunay tetrahedra.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## First development target

Start with the Swiss-cheese mocks:

1. geometry mock: `A` outside true spheres, `B` inside true spheres;
2. veto mock: `A` outside true spheres, `B` inside true spheres plus sparse exterior decoys.

The geometry mock validates the alpha shape, volume, and centroid. The veto mock validates the barrier-aware graph.

## Suggested first Copilot task

Ask Copilot to implement the modules in this order:

1. `src/paired_void_finder/periodic.py`
2. `src/paired_void_finder/mocks.py`
3. `src/paired_void_finder/graph.py`
4. `src/paired_void_finder/veto.py`
5. `src/paired_void_finder/boundaries.py`
6. `src/paired_void_finder/alpha_shape.py`
7. `src/paired_void_finder/voids.py`
8. `src/paired_void_finder/validation.py`

Do not implement Pinocchio readers yet. Keep the first version focused on correctness and the mock validation.
