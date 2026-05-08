"""Core data structures for the paired void finder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml


@dataclass(slots=True)
class Catalog:
    """Point catalog in a periodic cubic box."""

    positions: np.ndarray
    masses: np.ndarray | None
    box_size: float
    name: str = "catalog"

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.masses is not None:
            self.masses = np.asarray(self.masses, dtype=float)
            if self.masses.shape != (len(self.positions),):
                raise ValueError("masses must have shape (N,)")
        if self.box_size <= 0:
            raise ValueError("box_size must be positive")
        # Wrap positions into [0, box_size) so that cKDTree(boxsize=...) gets
        # valid inputs.  Floating-point boundary points from Pinocchio or other
        # codes that land exactly on box_size are also corrected here.
        self.positions = self.positions % self.box_size


@dataclass(slots=True)
class MockCatalog:
    """Synthetic A/B mock with known truth."""

    A: Catalog
    B: Catalog
    true_void_centers: np.ndarray
    true_void_radii: np.ndarray
    true_void_volumes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


_VALID_BOUNDARY_MODES: frozenset[str] = frozenset({"veto", "shell", "hybrid"})


@dataclass(slots=True)
class FinderParameters:
    """Parameters controlling the void finder."""

    M_A_min: float = 0.0
    M_B_min: float = 0.0
    b_BB: float = 1.5
    eta: float = 0.5
    N_veto: int = 8
    # boundary_mode selects how the A-boundary of each component is built:
    #   "veto"   – A points that vetoed inter-component B--B links (original behaviour)
    #   "shell"  – A points within b_shell * mean_A_spacing of any B point in the component
    #   "hybrid" – veto first; fall back to shell when |veto boundary| < N_A_min
    boundary_mode: str = "hybrid"
    b_shell: float = 1.5
    b_grow: float = 0.5
    lambda_alpha: float = 2.0
    N_B_min: int = 5
    N_A_min: int = 12
    R_min: float = 0.0
    enable_veto: bool = True

    def __post_init__(self) -> None:
        if self.boundary_mode not in _VALID_BOUNDARY_MODES:
            raise ValueError(
                f"boundary_mode must be one of {sorted(_VALID_BOUNDARY_MODES)!r}, "
                f"got {self.boundary_mode!r}"
            )
        if not self.enable_veto and self.boundary_mode == "veto":
            raise ValueError(
                "boundary_mode='veto' requires enable_veto=True. "
                "Use boundary_mode='shell' or boundary_mode='hybrid' instead."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinderParameters":
        """Load parameters from a YAML file with nested or flat keys."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        flat: dict[str, Any] = {}
        flat.update(raw.get("mass_cuts", {}))
        flat.update(raw.get("graph", {}))
        flat.update(raw.get("veto", {}))
        flat.update(raw.get("boundary", {}))
        flat.update(raw.get("alpha_shape", {}))
        flat.update(raw.get("selection", {}))
        flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})

        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in flat.items() if k in valid}
        return cls(**kwargs)


@dataclass(slots=True)
class Edge:
    """Candidate B--B graph edge."""

    i: int
    j: int
    accepted: bool = True
    veto_halos: tuple[int, ...] = ()


@dataclass(slots=True)
class AlphaShapeResult:
    """Output of a 3D alpha-shape calculation."""

    vertices: np.ndarray
    tetrahedra: np.ndarray
    accepted_tetrahedra: np.ndarray
    volume: float
    centroid_unwrapped: np.ndarray
    centroid_wrapped: np.ndarray
    effective_radius: float
    n_tetrahedra_total: int      # total Delaunay tetrahedra before alpha filtering
    n_tetrahedra_accepted: int   # tetrahedra surviving circumsphere and volume cuts
    alpha_fraction: float        # n_tetrahedra_accepted / n_tetrahedra_total


@dataclass(slots=True)
class Void:
    """Recovered void."""

    void_id: int
    center: np.ndarray
    volume: float
    effective_radius: float
    B_indices: np.ndarray
    A_boundary_indices: np.ndarray
    alpha_shape: AlphaShapeResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinderRun:
    """Diagnostic record from a single run_void_finder call.

    Index semantics
    ---------------
    Two index "spaces" are used throughout:

    *original indices*: into A.positions / B.positions **before** mass cuts.
    *sub-indices*: into A_pos / B_pos, the filtered sub-arrays after mass cuts.

    Conversion: ``A.positions[A_orig_indices[k]]`` is the position of
    sub-array A point *k*; ``B.positions[B_orig_indices[k]]`` likewise.

    Field-level notes:

    * ``A_orig_indices``, ``B_orig_indices`` – **original indices** of the
      points that survived the mass cut.
    * ``component_labels`` – one value per sub-array B point (length =
      len(B_orig_indices)).  Value is the component ID (0-based int).
    * ``veto_radii`` – one radius per sub-array A point (length =
      len(A_orig_indices)).  None when enable_veto=False.
    * ``edges`` – each Edge.i, Edge.j are **B sub-indices**.
    * ``*_boundary_sets`` – dict mapping component ID → numpy array of
      **A sub-indices**.  Use ``A_orig_indices[v]`` to convert to original
      catalog indices.  (``Void.A_boundary_indices`` already stores original
      catalog indices.)
    * ``voids`` – each ``Void.B_indices`` and ``Void.A_boundary_indices``
      store **original catalog indices**.
    """

    params: FinderParameters
    A_orig_indices: np.ndarray          # original A indices surviving mass cut
    B_orig_indices: np.ndarray          # original B indices surviving mass cut
    edges: list[Edge]                   # all B--B edges after veto annotation; i,j are B sub-indices
    veto_radii: np.ndarray | None       # per-sub-array-A veto radius; None if veto disabled
    component_labels: np.ndarray        # per-sub-array-B component label (0-based int)
    veto_boundary_sets: dict[int, np.ndarray]     # comp_id → A sub-index array (veto mechanism)
    shell_boundary_sets: dict[int, np.ndarray]    # comp_id → A sub-index array (shell mechanism)
    selected_boundary_sets: dict[int, np.ndarray] # comp_id → A sub-index array (before dilation)
    final_boundary_sets: dict[int, np.ndarray]    # comp_id → A sub-index array (after dilation)
    voids: list[Void]
