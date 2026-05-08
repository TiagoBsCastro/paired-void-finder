"""Swiss-cheese mock generation."""

from __future__ import annotations

import numpy as np

from .catalogs import Catalog, MockCatalog
from .periodic import periodic_distance, wrap_points


def generate_random_void_spheres(
    n_holes: int,
    box_size: float,
    radii: float | np.ndarray,
    seed: int | None = None,
    min_separation_factor: float = 2.0,
    max_attempts: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Place *n_holes* non-overlapping spherical voids at random positions.

    Holes are placed sequentially; each candidate is accepted only when its
    periodic distance to every already-placed hole exceeds
    ``min_separation_factor * (R_i + R_j)``.

    Parameters
    ----------
    n_holes:
        Number of voids to generate.
    box_size:
        Side length of the periodic cubic box.
    radii:
        Either a scalar radius (applied to all holes) or a 1-D array of
        length *n_holes* giving a per-hole radius.
    seed:
        RNG seed for reproducibility.  ``None`` uses an unpredictable seed.
    min_separation_factor:
        Minimum centre-to-centre separation expressed as a multiple of the
        sum of the two radii.  The condition enforced is::

            d(i, j) >= min_separation_factor * (R_i + R_j)

        For two equal-radius holes the surface-to-surface gap is
        ``(min_separation_factor - 1) * 2 * R``.  At the default of 2.0
        the gap equals 2 R (one diameter).  Use a value closer to 1.0 for
        tighter packing — e.g. 1.2 gives a gap of ≈ 0.4 R.
    max_attempts:
        Maximum number of random candidates tried per hole before raising.

    Returns
    -------
    centers:
        Shape ``(n_holes, 3)`` array of void centres in ``[0, box_size)``.
    radii_out:
        Shape ``(n_holes,)`` array of void radii (broadcast from *radii*).

    Raises
    ------
    RuntimeError
        If a hole cannot be placed after *max_attempts* candidates.
    """
    rng = np.random.default_rng(seed)

    if np.isscalar(radii):
        radii_arr = np.full(n_holes, float(radii))
    else:
        radii_arr = np.asarray(radii, dtype=float)
        if radii_arr.shape != (n_holes,):
            raise ValueError(
                f"radii must be a scalar or a 1-D array of length n_holes={n_holes}, "
                f"got shape {radii_arr.shape}"
            )

    placed_centers: list[np.ndarray] = []
    placed_radii: list[float] = []

    for i in range(n_holes):
        ri = float(radii_arr[i])
        placed = False
        for _ in range(max_attempts):
            c = rng.uniform(0.0, box_size, 3)
            # Reject if c is too close to any already-placed hole.
            ok = True
            for cj, rj in zip(placed_centers, placed_radii):
                d = float(periodic_distance(c[np.newaxis], cj, box_size)[0])
                if d < min_separation_factor * (ri + rj):
                    ok = False
                    break
            if ok:
                placed_centers.append(c)
                placed_radii.append(ri)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place hole {i} (R={ri:.3f}) after {max_attempts} attempts. "
                f"The box may be too crowded (n_holes={n_holes}, box_size={box_size}, "
                f"min_separation_factor={min_separation_factor})."
            )

    return np.array(placed_centers), radii_arr


def _grid_points(n_points: int, box_size: float, rng: np.random.Generator, jitter: bool) -> np.ndarray:
    n_side = int(np.ceil(n_points ** (1.0 / 3.0)))
    coords = (np.arange(n_side) + 0.5) * box_size / n_side
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid[:n_points].copy()
    if jitter:
        cell = box_size / n_side
        points += rng.uniform(-0.35 * cell, 0.35 * cell, size=points.shape)
    return wrap_points(points, box_size)


def _random_exterior_points(
    rng: np.random.Generator,
    n: int,
    box_size: float,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Generate n random points strictly outside all void spheres (periodic)."""
    collected: list[np.ndarray] = []
    while len(collected) < n:
        batch = rng.uniform(0.0, box_size, size=(max(n * 6, 512), 3))
        outside = np.ones(len(batch), dtype=bool)
        for c, r in zip(centers, radii):
            outside &= periodic_distance(batch, c, box_size) > r
        collected.extend(batch[outside])
    return np.asarray(collected[:n], dtype=float)


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
        veto: same as geometry, plus independently sampled random exterior
              points added to B as decoys (they do **not** overlap with A).
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

    A_positions = points[~inside_any]
    B_positions = points[inside_any]
    n_decoy = 0

    if mode in {"veto", "hard"}:
        frac = exterior_b_fraction if mode == "veto" else max(exterior_b_fraction, 0.08)
        n_decoy = int(round(frac * int(np.sum(~inside_any))))
        if n_decoy > 0:
            # Decoys are independent random exterior points, NOT grid points,
            # so they have no overlap with A.
            decoy_positions = _random_exterior_points(rng, n_decoy, box_size, centers, radii)
            B_positions = np.vstack([B_positions, decoy_positions])

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
            "n_decoy_b": n_decoy,
            "truth_id_for_grid_points": truth_id,
        },
    )
