#!/usr/bin/env python
"""Create a Swiss-cheese mock and save it as NPZ."""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from paired_void_finder.mocks import make_swiss_cheese_mock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mock.npz")
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="YAML file with mock parameters (e.g. configs/mock_geometry.yaml). "
             "Command-line flags override config file values.",
    )
    parser.add_argument("--box-size", type=float, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--mode", choices=["geometry", "veto", "hard"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Defaults used when neither config nor CLI specifies a value.
    box_size = 100.0
    n_points = 30000
    mode = "geometry"
    seed = 1234
    centers_raw: list = [[50.0, 50.0, 50.0]]
    radii_raw: list = [15.0]
    exterior_b_fraction = 0.0

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        box_size = float(cfg.get("box_size", box_size))
        n_points = int(cfg.get("n_points", n_points))
        mode = str(cfg.get("mode", mode))
        seed = int(cfg.get("seed", seed))
        centers_raw = cfg.get("void_centers", centers_raw)
        radii_raw = cfg.get("void_radii", radii_raw)
        exterior_b_fraction = float(cfg.get("exterior_b_fraction", exterior_b_fraction))

    # CLI flags override config values.
    if args.box_size is not None:
        box_size = args.box_size
    if args.n_points is not None:
        n_points = args.n_points
    if args.mode is not None:
        mode = args.mode
    if args.seed is not None:
        seed = args.seed

    centers = np.array(centers_raw)
    radii = np.array(radii_raw)

    kwargs: dict = {}
    if exterior_b_fraction > 0:
        kwargs["exterior_b_fraction"] = exterior_b_fraction

    mock = make_swiss_cheese_mock(
        box_size=box_size,
        n_points=n_points,
        void_centers=centers,
        void_radii=radii,
        mode=mode,
        seed=seed,
        **kwargs,
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
