#!/usr/bin/env python
"""Plot basic mock diagnostics."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.catalogs import Catalog, MockCatalog, Void
from paired_void_finder.plotting import plot_xy_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True)
    parser.add_argument("--voids")
    parser.add_argument("--out", default="mock_xy.png")
    args = parser.parse_args()

    m = np.load(args.mock)
    box_size = float(m["box_size"])
    mock = MockCatalog(
        A=Catalog(m["A_positions"], m["A_masses"], box_size, name="A"),
        B=Catalog(m["B_positions"], m["B_masses"], box_size, name="B"),
        true_void_centers=m["true_void_centers"],
        true_void_radii=m["true_void_radii"],
        true_void_volumes=m["true_void_volumes"],
    )
    voids = []
    if args.voids:
        v = np.load(args.voids)
        voids = [
            Void(i, c, float(vol), float(r), np.array([], dtype=int), np.array([], dtype=int))
            for i, (c, vol, r) in enumerate(zip(v["centers"], v["volumes"], v["effective_radii"]))
        ]
    plot_xy_projection(mock, voids, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
