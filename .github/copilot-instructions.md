# Copilot Instructions – paired-void-finder

## Build, test, and lint

```bash
# Install (editable, includes pytest + ruff)
pip install -e .[dev]

# Run all tests
pytest

# Run a single test
pytest tests/test_periodic.py::test_periodic_delta_across_boundary

# Lint
ruff check src tests scripts
```

Line length is 100 (configured in `pyproject.toml`).

## Architecture overview

The package is a **barrier-aware paired-catalog void finder**. Two catalogs — `A` (barriers) and `B` (interior tracers) — live in a periodic cubic box. The pipeline, implemented in `voids.run_void_finder`, proceeds in six stages:

1. **Graph** (`graph.py`): Build candidate B–B FOF edges with linking length `l_BB = b_BB × mean_B_spacing` using a periodic `cKDTree`.
2. **Veto** (`veto.py`): Reject edges whose segment passes within `r_veto,k = eta × d_N,k^A` of any A point. A two-pass strategy is used: broad KDTree radius query, then exact `distance_point_to_segment_periodic`.
3. **Components** (`boundaries.py`): Run union-find (implemented inline, not via networkx) on accepted edges to get connected-component labels.
4. **Boundary extraction** (`boundaries.py`): The A-boundary of each component is the union of vetoing A points on **inter-component** rejected edges only. Optionally dilated by adding A points within `l_grow = b_grow × mean_A_spacing`.
5. **Alpha shape** (`alpha_shape.py`): 3D alpha shape built from the A-boundary points (not B points). Points are unwrapped around a reference before Delaunay. Tetrahedra with circumsphere radius < `R_alpha = lambda_alpha × mean_A_spacing` are kept.
6. **Void catalog** (`voids.py`): Volume and volume-weighted centroid from accepted tetrahedra. Only components passing `N_B_min`, `N_A_min`, and `R_min` cuts are kept.

All data structures (`Catalog`, `MockCatalog`, `FinderParameters`, `Edge`, `AlphaShapeResult`, `Void`, `ValidationSummary`) are `@dataclass(slots=True)` and live in `catalogs.py`.

## Key conventions

### Periodic boundary conditions
- **Every** distance or nearest-neighbour query uses periodic BCs.
- Use `periodic_delta` / `periodic_distance` from `periodic.py` for point-to-point work.
- Pass `boxsize=box_size` to all `cKDTree` constructors.
- `alpha_shape_3d` unwraps its input with `unwrap_points` around a `reference` point before calling `Delaunay`; the output `centroid_unwrapped` is in unwrapped space, `centroid_wrapped = centroid_unwrapped % box_size`.

### Config YAML layout
`FinderParameters.from_yaml` reads nested YAML (`mass_cuts`, `graph`, `veto`, `boundary`, `alpha_shape`, `selection`) and flattens the keys. Add new parameters under the matching section. See `configs/algorithm_default.yaml` for the canonical structure.

### NPZ I/O schema
Mock and catalog NPZ files must provide:
- `A_positions` (N, 3), `A_masses` (N,)
- `B_positions` (N, 3), `B_masses` (N,)
- `box_size` (scalar)

### Mocks
`mocks.make_swiss_cheese_mock` supports three modes:
- `geometry` — A outside spheres, B strictly inside; used to validate alpha shape and centroid.
- `veto` — same plus sparse exterior B decoys; used to validate the barrier veto.
- `hard` — like `veto` with a larger decoy fraction.

### Test layout
Each test file covers one concern: `test_periodic.py`, `test_mock.py`, `test_veto.py`, `test_alpha_shape.py`, `test_pipeline_mock.py`. Keep this mapping; add a new file for each new module.

### Acceptance criteria (from AGENTS.md)
- Periodic distances correct across the box boundary.
- One-sphere geometry mock → one void, center error < `0.2 R_true`.
- Multi-sphere mock → approximately correct void count.
- Veto mock → less over-merging with veto enabled than disabled.
- Periodic-boundary mock → one void across the boundary, not two.

### Scripts
`scripts/` contains standalone CLI tools (`make_mock.py`, `run_void_finder.py`, `validate_mock.py`, `plot_mock_diagnostics.py`) that are separate from the package and import from `paired_void_finder`. Run them from the repo root after `pip install -e .`.

### Not yet implemented
Pinocchio catalog reading — a placeholder interface only. Do not implement in the first pass.
