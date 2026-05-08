"""Swiss-cheese mock generation."""

from __future__ import annotations

import numpy as np

from .catalogs import Catalog, MockCatalog
from .periodic import periodic_distance, wrap_points


def _grid_points(n_points: int, box_size: float, rng: np.random.Generator, jitter: bool) -> np.ndarray:
    n_side = int(np.ceil(n_points ** (1.0 / 3.0)))
    coords = (np.arange(n_side) + 0.5) * box_size / n_side
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid[:n_points].copy()
    if jitter:
        cell = box_size / n_side
        points += rng.uniform(-0.35 * cell, 0.35 * cell, size=points.shape)
    return wrap_points(points, box_size)


def make_swiss_cheese_mock(
    box_size: float,
    n_points: int,
    void_centers: np.ndarray,
    void_radii: np.ndarray,
    mode: str = "geometry",
    seed: int = 1234,
    jitter_grid: bool = True,
    exterior_b_fraction: float = 0.02,
) -> MockCatalog:
    """Create a synthetic Swiss-cheese A/B mock.

    Modes:
        geometry: A outside spheres, B inside spheres.
        veto: same as geometry, plus a sparse random subset of exterior points
              moved to B as decoys.
        hard: alias for veto with a larger exterior B fraction.
    """
    rng = np.random.default_rng(seed)
    centers = np.asarray(void_centers, dtype=float)
    radii = np.asarray(void_radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("void_centers must have shape (Nvoid, 3)")
    if radii.shape != (len(centers),):
        raise ValueError("void_radii must have shape (Nvoid,)")

    points = _grid_points(n_points, box_size, rng, jitter_grid)
    inside_any = np.zeros(n_points, dtype=bool)
    truth_id = -np.ones(n_points, dtype=int)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        inside = periodic_distance(points, center, box_size) <= radius
        assign = inside & ~inside_any
        truth_id[assign] = i
        inside_any |= inside

    if mode not in {"geometry", "veto", "hard"}:
        raise ValueError("mode must be 'geometry', 'veto', or 'hard'")

    B_mask = inside_any.copy()
    if mode in {"veto", "hard"}:
        frac = exterior_b_fraction if mode == "veto" else max(exterior_b_fraction, 0.08)
        exterior = np.flatnonzero(~inside_any)
        n_decoy = int(round(frac * len(exterior)))
        if n_decoy > 0:
            decoys = rng.choice(exterior, size=n_decoy, replace=False)
            B_mask[decoys] = True

    A_mask = ~inside_any
    A_positions = points[A_mask]
    B_positions = points[B_mask]
    A = Catalog(A_positions, np.ones(len(A_positions)), box_size, name="A")
    B = Catalog(B_positions, np.ones(len(B_positions)), box_size, name="B")
    true_volumes = 4.0 * np.pi * radii**3 / 3.0
    return MockCatalog(
        A=A,
        B=B,
        true_void_centers=wrap_points(centers, box_size),
        true_void_radii=radii,
        true_void_volumes=true_volumes,
        metadata={
            "mode": mode,
            "seed": seed,
            "n_points": n_points,
            "truth_id_for_base_points": truth_id,
        },
    )
