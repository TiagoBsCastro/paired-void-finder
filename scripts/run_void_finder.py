#!/usr/bin/env python
"""Run the void finder on an NPZ mock."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.catalogs import Catalog, FinderParameters
from paired_void_finder.voids import run_void_finder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True)
    parser.add_argument("--config", default="configs/algorithm_default.yaml")
    parser.add_argument("--out", default="voids.npz")
    args = parser.parse_args()

    data = np.load(args.mock)
    box_size = float(data["box_size"])
    A = Catalog(data["A_positions"], data["A_masses"], box_size, name="A")
    B = Catalog(data["B_positions"], data["B_masses"], box_size, name="B")
    params = FinderParameters.from_yaml(args.config)
    voids = run_void_finder(A, B, params)
    np.savez(
        args.out,
        centers=np.asarray([v.center for v in voids]) if voids else np.empty((0, 3)),
        volumes=np.asarray([v.volume for v in voids]),
        effective_radii=np.asarray([v.effective_radius for v in voids]),
    )
    print(f"Recovered {len(voids)} voids")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
