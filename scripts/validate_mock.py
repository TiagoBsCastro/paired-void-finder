#!/usr/bin/env python
"""Validate a recovered void catalog against mock truth."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.catalogs import AlphaShapeResult, Catalog, MockCatalog, Void
from paired_void_finder.validation import validate_against_mock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True)
    parser.add_argument("--voids", required=True)
    args = parser.parse_args()

    m = np.load(args.mock)
    v = np.load(args.voids)
    box_size = float(m["box_size"])
    mock = MockCatalog(
        A=Catalog(m["A_positions"], m["A_masses"], box_size, name="A"),
        B=Catalog(m["B_positions"], m["B_masses"], box_size, name="B"),
        true_void_centers=m["true_void_centers"],
        true_void_radii=m["true_void_radii"],
        true_void_volumes=m["true_void_volumes"],
    )
    voids = [
        Void(
            void_id=i,
            center=center,
            volume=float(volume),
            effective_radius=float(radius),
            B_indices=np.array([], dtype=int),
            A_boundary_indices=np.array([], dtype=int),
            alpha_shape=None,
        )
        for i, (center, volume, radius) in enumerate(
            zip(v["centers"], v["volumes"], v["effective_radii"])
        )
    ]
    summary = validate_against_mock(voids, mock)
    print(summary)


if __name__ == "__main__":
    main()
