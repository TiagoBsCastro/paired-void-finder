#!/usr/bin/env python
"""Run the void finder on a Swiss-cheese mock and save full diagnostic outputs.

Usage
-----
    python scripts/run_swiss_cheese_diagnostics.py \\
        --mock-config configs/mock_geometry.yaml \\
        --finder-config configs/algorithm_default.yaml \\
        --outdir results/geometry_run

Outputs saved to --outdir
-------------------------
    void_catalog.npz            – recovered void centres, volumes, radii
    run_diagnostics.npz         – edges, labels, boundary sets
    summary.txt                 – human-readable validation summary
    match_table.csv             – per-matched-pair metrics
    xy_projection.png
    slice_z.png
    3d_truth_recovered.png
    radial_profile.png
    component_size_dist.png
    boundary_size_dist.png
    alpha_diagnostics.png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import yaml


def _ragged_boundary(
    d: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialise a ``{comp_id: A_indices}`` dict to three flat arrays."""
    if not d:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    comp_ids = np.array(sorted(d.keys()), dtype=int)
    counts = np.array([len(d[c]) for c in comp_ids], dtype=int)
    flat = (
        np.concatenate([d[c] for c in comp_ids])
        if counts.sum() > 0
        else np.array([], dtype=int)
    )
    return comp_ids, counts, flat


def _save_run_diagnostics(path: Path, run, voids: list) -> None:
    edges = run.edges
    if edges:
        edge_i = np.array([e.i for e in edges], dtype=int)
        edge_j = np.array([e.j for e in edges], dtype=int)
        edge_accepted = np.array([e.accepted for e in edges], dtype=bool)
        halo_counts = np.array([len(e.veto_halos) for e in edges], dtype=int)
        halos_flat = (
            np.concatenate([np.asarray(e.veto_halos, dtype=int)
                            for e in edges if len(e.veto_halos) > 0])
            if any(len(e.veto_halos) > 0 for e in edges)
            else np.array([], dtype=int)
        )
    else:
        edge_i = edge_j = np.array([], dtype=int)
        edge_accepted = np.array([], dtype=bool)
        halo_counts = halos_flat = np.array([], dtype=int)

    veto_ids, veto_counts, veto_flat = _ragged_boundary(run.veto_boundary_sets)
    shell_ids, shell_counts, shell_flat = _ragged_boundary(run.shell_boundary_sets)
    sel_ids, sel_counts, sel_flat = _ragged_boundary(run.selected_boundary_sets)
    fin_ids, fin_counts, fin_flat = _ragged_boundary(run.final_boundary_sets)

    n_accepted = int(edge_accepted.sum()) if len(edge_accepted) else 0
    n_rejected = int((~edge_accepted).sum()) if len(edge_accepted) else 0
    comp_ids = np.unique(run.component_labels)

    np.savez(
        path,
        A_orig_indices=run.A_orig_indices,
        B_orig_indices=run.B_orig_indices,
        component_labels=run.component_labels,
        veto_radii=run.veto_radii if run.veto_radii is not None else np.array([]),
        edge_i=edge_i,
        edge_j=edge_j,
        edge_accepted=edge_accepted,
        edge_veto_halo_counts=halo_counts,
        edge_veto_halos_flat=halos_flat,
        n_edges_accepted=np.array([n_accepted]),
        n_edges_rejected=np.array([n_rejected]),
        n_components=np.array([len(comp_ids)]),
        n_voids=np.array([len(voids)]),
        veto_boundary_comp_ids=veto_ids,
        veto_boundary_counts=veto_counts,
        veto_boundary_flat=veto_flat,
        shell_boundary_comp_ids=shell_ids,
        shell_boundary_counts=shell_counts,
        shell_boundary_flat=shell_flat,
        selected_boundary_comp_ids=sel_ids,
        selected_boundary_counts=sel_counts,
        selected_boundary_flat=sel_flat,
        final_boundary_comp_ids=fin_ids,
        final_boundary_counts=fin_counts,
        final_boundary_flat=fin_flat,
    )
    print(f"  edges accepted/rejected: {n_accepted}/{n_rejected}")
    print(f"  components: {len(comp_ids)}, final voids: {len(voids)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run void finder on a Swiss-cheese mock and save diagnostic outputs."
    )
    parser.add_argument(
        "--mock-config", required=True, metavar="YAML",
        help="YAML file describing the Swiss-cheese mock (box_size, void_centers, …)",
    )
    parser.add_argument(
        "--finder-config",
        default=os.path.join(os.path.dirname(__file__), "..", "configs",
                             "algorithm_default.yaml"),
        metavar="YAML",
        help="YAML file with FinderParameters (default: configs/algorithm_default.yaml).",
    )
    parser.add_argument(
        "--outdir", default=".", metavar="DIR",
        help="Directory where all outputs are saved (created if absent).",
    )
    # ── Random-holes overrides ────────────────────────────────────────────────
    parser.add_argument(
        "--random-holes", action="store_true",
        help="Generate random void centres instead of using void_centers from YAML.",
    )
    parser.add_argument(
        "--n-holes", type=int, default=None, metavar="N",
        help="Number of randomly placed voids (implies --random-holes; requires --hole-radius).",
    )
    parser.add_argument(
        "--hole-radius", type=float, default=None, metavar="R",
        help="Radius applied to all randomly generated voids.",
    )
    parser.add_argument(
        "--min-separation-factor", type=float, default=2.0, metavar="F",
        help="Minimum centre separation as a multiple of (R_i + R_j). Default: 2.0.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="RNG seed (overrides the YAML seed value).",
    )
    args = parser.parse_args()

    # Validate random-holes argument combinations.
    use_random = args.random_holes or (args.n_holes is not None)
    if use_random:
        if args.n_holes is None:
            parser.error("--n-holes is required when --random-holes is used")
        if args.hole_radius is None:
            parser.error("--hole-radius is required when --random-holes is used")

    from paired_void_finder.catalogs import FinderParameters
    from paired_void_finder.diagnostics import (
        match_voids_to_truth,
        plot_3d_truth_and_recovered,
        plot_alpha_diagnostics,
        plot_boundary_size_distribution,
        plot_component_size_distribution,
        plot_radial_profile,
        plot_slice_truth_vs_found,
        plot_xy_projection,
    )
    from paired_void_finder.mocks import generate_random_void_spheres, make_swiss_cheese_mock
    from paired_void_finder.voids import run_void_finder

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load mock config and generate mock ──────────────────────────────────
    with open(args.mock_config, "r", encoding="utf-8") as f:
        mock_cfg = yaml.safe_load(f) or {}

    box_size = float(mock_cfg.get("box_size", 100.0))
    n_points = int(mock_cfg.get("n_points", 30000))
    mode = str(mock_cfg.get("mode", "geometry"))
    seed = int(mock_cfg.get("seed", 1234))
    void_centers = np.array(mock_cfg.get("void_centers", [[50.0, 50.0, 50.0]]))
    void_radii = np.array(mock_cfg.get("void_radii", [15.0]))
    exterior_b_fraction = float(mock_cfg.get("exterior_b_fraction", 0.0))

    # CLI overrides.
    if args.seed is not None:
        seed = args.seed
    if use_random:
        void_centers, void_radii = generate_random_void_spheres(
            n_holes=args.n_holes,
            box_size=box_size,
            radii=args.hole_radius,
            seed=seed,
            min_separation_factor=args.min_separation_factor,
        )
        print(
            f"Generated {args.n_holes} random holes "
            f"(R={args.hole_radius}, min_sep={args.min_separation_factor})"
        )
    if len(void_centers) != len(void_radii):
        raise ValueError(
            f"len(void_centers)={len(void_centers)} != len(void_radii)={len(void_radii)}"
        )

    kw: dict = {}
    if exterior_b_fraction > 0.0:
        kw["exterior_b_fraction"] = exterior_b_fraction
    mock = make_swiss_cheese_mock(
        box_size=box_size,
        n_points=n_points,
        void_centers=void_centers,
        void_radii=void_radii,
        mode=mode,
        seed=seed,
        **kw,
    )
    print(
        f"Mock: {len(mock.A.positions)} A points, {len(mock.B.positions)} B points, "
        f"{len(mock.true_void_radii)} true voids"
    )

    # ── Save the final mock config used ─────────────────────────────────────
    mock_used_cfg: dict = {
        "box_size": box_size,
        "n_points": n_points,
        "mode": mode,
        "seed": seed,
        "void_centers": void_centers.tolist(),
        "void_radii": void_radii.tolist(),
    }
    if exterior_b_fraction > 0.0:
        mock_used_cfg["exterior_b_fraction"] = exterior_b_fraction
    with open(outdir / "mock_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(mock_used_cfg, f, default_flow_style=None, sort_keys=False)

    # ── Run void finder ──────────────────────────────────────────────────────
    params = FinderParameters.from_yaml(args.finder_config)
    voids, run = run_void_finder(mock.A, mock.B, params, return_diagnostics=True)  # type: ignore[misc]
    print(f"Recovered {len(voids)} voids")

    # ── Validate ─────────────────────────────────────────────────────────────
    summary = match_voids_to_truth(voids, mock)
    print(
        f"Matched {summary.n_matched}/{summary.n_true}, "
        f"missed {summary.n_missed}, FP {summary.false_positive_count}"
    )

    # ── void_catalog.npz ─────────────────────────────────────────────────────
    np.savez(
        outdir / "void_catalog.npz",
        centers=(
            np.asarray([v.center for v in voids]) if voids else np.empty((0, 3))
        ),
        volumes=np.asarray([v.volume for v in voids]) if voids else np.array([]),
        effective_radii=(
            np.asarray([v.effective_radius for v in voids]) if voids else np.array([])
        ),
    )

    # ── run_diagnostics.npz ──────────────────────────────────────────────────
    _save_run_diagnostics(outdir / "run_diagnostics.npz", run, voids)

    # ── summary.txt ──────────────────────────────────────────────────────────
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"n_true: {summary.n_true}\n")
        f.write(f"n_recovered: {summary.n_recovered}\n")
        f.write(f"n_matched: {summary.n_matched}\n")
        f.write(f"n_missed: {summary.n_missed}\n")
        f.write(f"n_duplicate_matches: {summary.n_duplicate_matches}\n")
        f.write(f"false_positive_count: {summary.false_positive_count}\n")
        f.write(f"largest_volume_fraction: {summary.largest_volume_fraction:.6f}\n")
        if len(summary.center_errors) > 0:
            f.write(f"mean_center_error: {float(np.mean(summary.center_errors)):.4f}\n")
            f.write(f"mean_radius_error: {float(np.mean(summary.radius_errors)):.4f}\n")
            f.write(f"mean_volume_error: {float(np.mean(summary.volume_errors)):.4f}\n")

    # ── match_table.csv ──────────────────────────────────────────────────────
    with open(outdir / "match_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "true_id", "void_id", "center_error", "radius_error", "volume_error",
            "center_x", "center_y", "center_z", "R_eff", "volume",
            "N_B", "N_A_boundary",
            "alpha_tetrahedra_total", "alpha_tetrahedra_accepted", "alpha_fraction",
            "component_id",
        ])
        for true_id, void_id, ce, re, ve in zip(
            summary.matched_true_indices,
            summary.matched_void_indices,
            summary.center_errors,
            summary.radius_errors,
            summary.volume_errors,
        ):
            v = voids[int(void_id)]
            cx, cy, cz = v.center
            if v.alpha_shape is not None:
                n_tot = v.alpha_shape.n_tetrahedra_total
                n_acc = v.alpha_shape.n_tetrahedra_accepted
                afrac = v.alpha_shape.alpha_fraction
            else:
                n_tot = n_acc = afrac = ""
            comp_id = v.metadata.get("component_id", "")
            writer.writerow([
                int(true_id), int(void_id), f"{ce:.4f}", f"{re:.4f}", f"{ve:.4f}",
                f"{cx:.4f}", f"{cy:.4f}", f"{cz:.4f}",
                f"{v.effective_radius:.4f}", f"{v.volume:.4f}",
                len(v.B_indices), len(v.A_boundary_indices),
                n_tot, n_acc,
                f"{afrac:.4f}" if isinstance(afrac, float) else afrac,
                comp_id,
            ])

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    plot_xy_projection(mock, voids, summary, outpath=outdir / "xy_projection.png")
    plot_slice_truth_vs_found(
        mock, voids, summary, outpath=outdir / "slice_z.png", true_id=0, axis="z"
    )
    plot_3d_truth_and_recovered(
        mock, voids, summary, outpath=outdir / "3d_truth_recovered.png", true_id=0
    )
    plot_radial_profile(
        mock, voids, summary, outpath=outdir / "radial_profile.png", true_id=0
    )
    plot_radial_profile(
        mock, voids, summary, outpath=outdir / "radial_profile_normalized.png",
        true_id=0, normalize_by_mean=True,
    )
    plot_component_size_distribution(run, outpath=outdir / "component_size_dist.png")
    plot_boundary_size_distribution(run, outpath=outdir / "boundary_size_dist.png")
    plot_alpha_diagnostics(voids, summary, outpath=outdir / "alpha_diagnostics.png")

    print(f"Outputs saved to {outdir}")


if __name__ == "__main__":
    main()
