# Paired Void Finder

Barrier-aware paired-catalog void finder in a periodic cubic box.

Given two point catalogs **A** (barriers) and **B** (void tracers), the algorithm:

1. Builds a B–B friends-of-friends graph.
2. Vetoes links where the segment passes too close to any A point.
3. Finds connected components of the accepted graph (protovoids).
4. Builds an A boundary for each component (veto, shell, or hybrid mode).
5. Optionally dilates the boundary with nearby A points.
6. Computes a 3D alpha shape, volume, and volume centroid.

## Install

```bash
conda create -n pvf python=3.11
conda activate pvf
pip install -e .[dev]
pytest
```

## Quick start

```python
import numpy as np
from paired_void_finder.catalogs import Catalog, FinderParameters
from paired_void_finder.voids import run_void_finder

box = 100.0
A = Catalog(np.random.uniform(0, box, (5000, 3)), None, box, name="A")
B = Catalog(np.random.uniform(0, box, (1000, 3)), None, box, name="B")

params = FinderParameters(boundary_mode="hybrid", lambda_alpha=2.0)
voids = run_void_finder(A, B, params)
for v in voids:
    print(f"center={v.center}, R={v.effective_radius:.2f}, V={v.volume:.2f}")
```

## Boundary modes

The `boundary_mode` parameter controls how the A-boundary of each protovoid
component is defined.

| Mode | Description |
|------|-------------|
| `"veto"` | A points that vetoed inter-component B–B links. Requires `enable_veto=True`. Accurate when barriers are dense, but may be sparse in low-density regions. |
| `"shell"` | A points within `b_shell × mean_A_spacing` of any B member. More robust for isolated voids and works without veto (`enable_veto=False`). |
| `"hybrid"` | **(default)** Tries veto first; falls back to shell if `len(veto_boundary) < N_A_min`. Combines the accuracy of veto with the reliability of shell. |

The guard `enable_veto=False` together with `boundary_mode="veto"` raises
`ValueError` at construction time:

```python
# This raises ValueError:
FinderParameters(enable_veto=False, boundary_mode="veto")
```

### Algorithm parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `b_BB` | 1.5 | FOF linking length factor (times mean B spacing) |
| `eta` | 0.5 | Veto radius scale factor |
| `N_veto` | 8 | Nearest-A-neighbor rank used for local veto radius |
| `b_shell` | 1.5 | Shell boundary search radius factor |
| `b_grow` | 0.5 | Boundary dilation factor |
| `lambda_alpha` | 2.0 | Alpha-shape circumsphere radius factor |
| `N_B_min` | 5 | Minimum B members per void |
| `N_A_min` | 12 | Minimum A boundary points (also hybrid fallback threshold) |
| `R_min` | 0.0 | Minimum effective radius |
| `enable_veto` | True | Whether to apply the veto step |
| `boundary_mode` | `"hybrid"` | Boundary construction mode |

Parameters can also be loaded from a YAML file:

```bash
# configs/algorithm_default.yaml
python scripts/run_void_finder.py --mock mock.npz --config configs/algorithm_default.yaml --out voids.npz
```

## Mock catalogs and YAML configs

Two YAML-driven mock configs are provided in `configs/`:

**`configs/mock_geometry.yaml`** — geometry mock (A outside spheres, B inside):
```yaml
box_size: 100.0
n_points: 30000
mode: geometry
seed: 1234
void_centers:
  - [50.0, 50.0, 50.0]
void_radii: [15.0]
```

**`configs/mock_veto.yaml`** — veto mock (adds exterior B decoys):
```yaml
box_size: 100.0
n_points: 30000
mode: veto
seed: 1234
exterior_b_fraction: 0.02
void_centers:
  - [35.0, 50.0, 50.0]
  - [65.0, 50.0, 50.0]
void_radii: [12.0, 12.0]
```

### `scripts/make_mock.py` — generate a mock NPZ

```bash
python scripts/make_mock.py --config configs/mock_geometry.yaml --out mock_geometry.npz
```

The following CLI flags override the corresponding YAML keys:

| Flag | YAML key | Description |
|------|----------|-------------|
| `--box-size` | `box_size` | Cubic box side length |
| `--n-points` | `n_points` | Total tracer count (split between A and B) |
| `--seed` | `seed` | RNG seed |
| `--mode` | `mode` | Mock mode: `geometry`, `veto`, or `hard` |

## Swiss-cheese diagnostics script

### `scripts/run_swiss_cheese_diagnostics.py` — run and diagnose

Run a full diagnostic pass on a YAML-defined mock:

```bash
python scripts/run_swiss_cheese_diagnostics.py \
    --mock-config configs/mock_geometry.yaml \
    --finder-config configs/algorithm_default.yaml \
    --outdir results/geometry_run
```

Required / standard flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--mock-config` | (required) | YAML file describing the mock |
| `--finder-config` | `configs/algorithm_default.yaml` | YAML file with `FinderParameters` |
| `--outdir` | `.` | Output directory (created if absent) |

### Outputs

| File | Description |
|------|-------------|
| `void_catalog.npz` | Recovered void centres, volumes, radii |
| `run_diagnostics.npz` | Edges, component labels, boundary sets |
| `summary.txt` | Human-readable validation summary |
| `match_table.csv` | Per-matched-pair metrics |
| `mock_used.yaml` | Final mock config used (incl. generated centres) |
| `all_voids_overview_xy.png` | XY overview of all true/recovered voids with labels |
| `xy_projection.png` | XY scatter of A/B + true circles + recovered centres |
| `slice_z_true_{N:03d}.png` | Per-hole 2D slice at the true centre; shows matched reconstruction if available |
| `3d_truth_recovered_true_{N:03d}.png` | Per-hole 3D alpha-shape vs true sphere |
| `radial_profile_true_{N:03d}.png` | Per-hole radial A/B density profiles |
| `radial_profile_normalized_true_{N:03d}.png` | Same, normalised by mean density |
| `slice_z_void_{N:03d}.png` | Per recovered void, slab at the recovered centre; always shows reconstruction |
| `slice_z.png` | Backward-compatible alias → `slice_z_true_000.png` |
| `3d_truth_recovered.png` | Backward-compatible alias → `3d_truth_recovered_true_000.png` |
| `radial_profile.png` | Backward-compatible alias → `radial_profile_true_000.png` |
| `radial_profile_normalized.png` | Backward-compatible alias → `radial_profile_normalized_true_000.png` |
| `component_size_dist.png` | B component size histogram |
| `boundary_size_dist.png` | A boundary size by pipeline stage |
| `alpha_diagnostics.png` | Alpha-shape tetrahedra counts and acceptance fraction |

### Random hole generation

Pass `--random-holes` (or just `--n-holes`) to override the `void_centers` and
`void_radii` from the YAML with randomly placed holes:

```bash
python scripts/run_swiss_cheese_diagnostics.py \
    --mock-config configs/mock_geometry.yaml \
    --finder-config configs/algorithm_default.yaml \
    --outdir results/random_5_holes \
    --random-holes --n-holes 5 --hole-radius 10 \
    --min-separation-factor 2.5 --seed 42
```

Holes are placed with rejection sampling in a periodic box.  The
non-overlap condition is `d(i,j) >= min_separation_factor * (R_i + R_j)`;
at the default of 2.0 the surface-to-surface gap equals one diameter for
equal-radius holes.

| Flag | Default | Description |
|------|---------|-------------|
| `--random-holes` | — | Enable random hole generation (overrides YAML `void_centers`). |
| `--n-holes N` | — | Number of holes (required with `--random-holes`). |
| `--hole-radius R` | — | Uniform radius for all holes (required with `--random-holes`). |
| `--min-separation-factor F` | 2.0 | `d >= F*(R_i+R_j)` non-overlap condition. Gap = `2*(F-1)*R` for equal radii. |
| `--seed S` | YAML value | RNG seed (overrides YAML `seed`). |

The final mock configuration used (including generated centres) is always saved to
`<outdir>/mock_used.yaml` for reproducibility.

## Diagnostics

Pass `return_diagnostics=True` to get a `FinderRun` dataclass alongside the void
list:

```python
voids, run = run_void_finder(A, B, params, return_diagnostics=True)
```

### FinderRun index semantics

Two index spaces are used:

- **Original indices**: into `A.positions` / `B.positions` before mass cuts.
- **Sub-indices**: into the filtered sub-arrays after mass cuts.

| Field | Dtype | Index space |
|-------|-------|-------------|
| `A_orig_indices` | int array | original A indices that survived mass cut |
| `B_orig_indices` | int array | original B indices that survived mass cut |
| `component_labels` | int array, len=len(B_orig_indices) | B sub-index → component ID |
| `veto_radii` | float array, len=len(A_orig_indices) | A sub-index → veto radius |
| `edges[k].i`, `edges[k].j` | int | B sub-indices |
| `veto_boundary_sets[comp]` | int array | **A sub-indices** |
| `shell_boundary_sets[comp]` | int array | **A sub-indices** |
| `selected_boundary_sets[comp]` | int array | **A sub-indices** (before dilation) |
| `final_boundary_sets[comp]` | int array | **A sub-indices** (after dilation) |
| `Void.B_indices` | int array | **original** catalog indices |
| `Void.A_boundary_indices` | int array | **original** catalog indices |

Convert A sub-indices to original catalog indices:
```python
orig_idx = run.A_orig_indices[run.final_boundary_sets[comp_id]]
```

### Saving diagnostics to NPZ

```bash
python scripts/run_void_finder.py --mock mock.npz --out voids.npz --diagnostics diag.npz
```

The diagnostics NPZ contains:
- `A_orig_indices`, `B_orig_indices`, `component_labels`, `veto_radii`
- `edge_i`, `edge_j`, `edge_accepted`, `edge_veto_halo_counts`, `edge_veto_halos_flat`
- Four boundary sets as ragged arrays: `{veto,shell,selected,final}_boundary_{comp_ids,counts,flat}`
- Summary scalars: `n_edges_accepted`, `n_edges_rejected`, `n_components`, `n_voids`

## Validation workflow

Use `validate_against_mock` when you have a `MockCatalog` with known sphere
positions and radii:

```python
from paired_void_finder.validation import validate_against_mock

voids = run_void_finder(mock.A, mock.B, params)
summary = validate_against_mock(voids, mock)

print(f"Matched: {summary.n_matched}/{summary.n_true}")
print(f"Missed:  {summary.n_missed}")
print(f"Duplicates: {summary.n_duplicate_matches}")
print(f"Center errors (d/R_true): {summary.center_errors}")
print(f"Radius errors (ΔR/R_true): {summary.radius_errors}")
print(f"Volume errors (ΔV/V_true): {summary.volume_errors}")
```

### ValidationSummary fields

| Field | Description |
|-------|-------------|
| `n_true` | Number of true spheres in mock |
| `n_recovered` | Number of recovered voids |
| `n_matched` | True spheres covered by a unique matched void (one-to-one) |
| `n_missed` | True spheres with no matched void |
| `n_duplicate_matches` | Recovered voids whose center falls inside an already-matched sphere |
| `matched_true_indices` | Sorted indices into `mock.true_void_centers` |
| `matched_void_indices` | Parallel indices into the recovered void list |
| `center_errors` | `d(recovered, true) / R_true` for each matched pair |
| `radius_errors` | `(R_rec − R_true) / R_true` for each matched pair |
| `volume_errors` | `(V_rec − V_true) / V_true` for each matched pair |
| `false_positive_count` | Recovered voids outside all true spheres |
| `largest_volume_fraction` | Largest recovered void volume / box volume |

Matching uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
for strict one-to-one assignment; each true sphere can only match at most one
recovered void.

## Running the tests

```bash
pytest -v
```

The test suite includes:
- Unit tests for periodic distances, FOF graph, veto, boundaries, alpha shape
- Pipeline mock tests with one-to-one recovery assertions (center, radius, volume)
- Acceptance tests: one-sphere, multi-sphere, veto, and periodic-boundary mocks
- Validation unit tests for LSA matching, duplicate detection, and edge cases

## Project layout

```
src/paired_void_finder/
  catalogs.py     – Dataclasses (Catalog, FinderParameters, Void, FinderRun, …)
  periodic.py     – Periodic distance utilities
  graph.py        – B–B FOF graph construction
  veto.py         – Veto radius computation and link filtering
  boundaries.py   – Veto, shell, and dilation boundary extraction
  alpha_shape.py  – 3D alpha shape, volume, centroid
  voids.py        – Top-level pipeline (run_void_finder)
  mocks.py        – Swiss-cheese mock generation (`make_swiss_cheese_mock`, `generate_random_void_spheres`)
  validation.py   – validate_against_mock, ValidationSummary
configs/
  algorithm_default.yaml
  mock_geometry.yaml
  mock_veto.yaml
scripts/
  make_mock.py          – Generate a mock NPZ from a YAML config
  run_void_finder.py    – Run the pipeline with optional diagnostics output
tests/
  test_acceptance.py    – End-to-end acceptance tests
  test_pipeline_mock.py – Pipeline recovery tests
  test_validation.py    – Validation metric unit tests
  test_*.py             – Unit tests per module
```

