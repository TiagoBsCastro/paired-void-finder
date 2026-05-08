#!/usr/bin/env python
"""Create a Swiss-cheese mock and save it as NPZ."""

from __future__ import annotations

import argparse
import numpy as np

from paired_void_finder.mocks import make_swiss_cheese_mock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mock.npz")
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--n-points", type=int, default=30000)
    parser.add_argument("--mode", choices=["geometry", "veto", "hard"], default="geometry")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    centers = np.array([[50.0, 50.0, 50.0]])
    radii = np.array([15.0])
    mock = make_swiss_cheese_mock(
        box_size=args.box_size,
        n_points=args.n_points,
        void_centers=centers,
        void_radii=radii,
        mode=args.mode,
        seed=args.seed,
    )
    np.savez(
        args.out,
        A_positions=mock.A.positions,
        B_positions=mock.B.positions,
        A_masses=mock.A.masses,
        B_masses=mock.B.masses,
        true_void_centers=mock.true_void_centers,
        true_void_radii=mock.true_void_radii,
        true_void_volumes=mock.true_void_volumes,
        box_size=mock.A.box_size,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
