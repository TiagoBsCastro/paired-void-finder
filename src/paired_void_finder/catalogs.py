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
    boundary_mode: str = "veto"
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

    All index arrays that reference A or B points use **original** catalog
    indices (i.e. into A.positions / B.positions before mass cuts).
    Sub-array indices (into the mass-cut-filtered positions) are used only
    internally and are not stored here.
    """

    params: FinderParameters
    A_orig_indices: np.ndarray              # original A indices surviving mass cut
    B_orig_indices: np.ndarray              # original B indices surviving mass cut
    edges: list[Edge]                       # all B--B edges after veto annotation
    veto_radii: np.ndarray | None           # per-masked-A veto radius; None if disabled
    component_labels: np.ndarray            # per-masked-B component label (0-based)
    veto_boundary_sets: dict[int, np.ndarray]   # A sub-indices before dilation
    final_boundary_sets: dict[int, np.ndarray]  # A sub-indices after dilation
    voids: list[Void]
