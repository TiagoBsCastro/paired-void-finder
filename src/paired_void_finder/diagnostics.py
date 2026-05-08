"""Diagnostic helpers and plots for Swiss-cheese void validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .catalogs import FinderRun, MockCatalog, Void
from .periodic import periodic_distance, unwrap_points
from .validation import ValidationSummary, validate_against_mock

# ── Non-plot helpers ──────────────────────────────────────────────────────────


def match_voids_to_truth(voids: list[Void], mock: MockCatalog) -> ValidationSummary:
    """Run one-to-one Hungarian matching of recovered voids against mock truth."""
    return validate_against_mock(voids, mock)


def select_best_match(
    voids: list[Void], mock: MockCatalog, true_id: int = 0
) -> tuple[Void | None, int | None]:
    """Return ``(void, void_index)`` for the recovered void matched to ``true_id``.

    Returns ``(None, None)`` when no recovered void was matched to ``true_id``.
    The ``void_index`` is the index into ``voids``.
    """
    if not voids:
        return None, None
    summary = validate_against_mock(voids, mock)
    matched_true = summary.matched_true_indices
    hits = np.where(matched_true == true_id)[0]
    if len(hits) == 0:
        return None, None
    void_idx = int(summary.matched_void_indices[hits[0]])
    return voids[void_idx], void_idx


def external_faces_from_tetrahedra(tetrahedra: np.ndarray) -> np.ndarray:
    """Return unique external (surface) triangular faces of a tetrahedral mesh.

    A face is external when it belongs to exactly one tetrahedron.

    Parameters
    ----------
    tetrahedra:
        Integer array of shape ``(N, 4)`` — vertex indices per tetrahedron.

    Returns
    -------
    faces:
        Integer array of shape ``(M, 3)`` with sorted vertex indices per row.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (N, 4)")
    # Each tetrahedron has 4 triangular faces.
    combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    all_faces = np.empty((len(tetrahedra) * 4, 3), dtype=int)
    for k, (a, b, c) in enumerate(combos):
        all_faces[k::4] = np.sort(tetrahedra[:, [a, b, c]], axis=1)
    unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)
    return unique_faces[counts == 1]


def radial_profile(
    points: np.ndarray,
    center: np.ndarray,
    box_size: float,
    r_bins: np.ndarray,
    normalize_by_mean: bool = False,
) -> np.ndarray:
    """Number density of ``points`` in spherical shells around ``center`` (periodic BCs).

    Parameters
    ----------
    points:
        Point positions, shape ``(N, 3)``.
    center:
        Reference position, shape ``(3,)``.
    box_size:
        Side length of the periodic box.
    r_bins:
        Bin edges, shape ``(nbins+1,)``.
    normalize_by_mean:
        When ``True``, divide each shell density by the mean catalog number density
        ``n_total / box_size**3``, so that the profile is dimensionless and equals
        1 in a uniform distribution.

    Returns
    -------
    density:
        Counts divided by shell volume (and optionally by mean density),
        shape ``(nbins,)``.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    r = periodic_distance(points, center, box_size)
    counts, _ = np.histogram(r, bins=r_bins)
    shell_volumes = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    density = np.where(shell_volumes > 0, counts / shell_volumes, 0.0)
    if normalize_by_mean and len(points) > 0:
        n_mean = len(points) / box_size ** 3
        density = density / n_mean
    return density


# ── Private plot utilities ────────────────────────────────────────────────────


def triangle_plane_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    axis_index: int,
    plane_value: float,
    atol: float = 1e-10,
) -> list[np.ndarray]:
    """Return line segments where a triangulated surface intersects a coordinate plane.

    Parameters
    ----------
    vertices:
        Vertex positions, shape ``(N, 3)``, in a consistent (unwrapped) frame.
    faces:
        Triangular faces, shape ``(M, 3)``, integer indices into *vertices*.
    axis_index:
        Index of the slice axis: 0 = x, 1 = y, 2 = z.
    plane_value:
        Coordinate value of the cutting plane along *axis_index*.
    atol:
        Vertices within *atol* of the plane are treated as lying on the plane.
        This avoids double-counting a segment when an edge is exactly in the plane.

    Returns
    -------
    segments:
        List of ``(2, 3)`` float arrays, one per intersecting triangle.
        Each row is one endpoint of the segment in 3D.
        Triangles that do not cross the plane produce no entry.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    segments: list[np.ndarray] = []
    for tri in faces:
        p = vertices[tri]               # (3, 3)
        d = p[:, axis_index] - plane_value
        # Snap near-plane vertices to exactly on the plane to avoid float noise.
        d = np.where(np.abs(d) <= atol, 0.0, d)
        crossings: list[np.ndarray] = []
        for i, j in ((0, 1), (1, 2), (0, 2)):
            if d[i] * d[j] < 0:        # strict sign change → edge crosses plane
                t = d[i] / (d[i] - d[j])
                crossings.append(p[i] + t * (p[j] - p[i]))
            elif d[i] == 0.0 and d[j] != 0.0:   # vertex exactly on plane
                crossings.append(p[i].copy())
        # Remove duplicates that arise when two edges share an on-plane vertex.
        unique: list[np.ndarray] = []
        for c in crossings:
            if not any(np.allclose(c, u, atol=atol) for u in unique):
                unique.append(c)
        if len(unique) == 2:
            segments.append(np.array([unique[0], unique[1]]))
    return segments


def _save_or_show(fig: plt.Figure, outpath: str | Path | None) -> None:
    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _legend_dedup(ax: plt.Axes) -> None:
    """Add legend to *ax*, removing duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_h: list = []
    unique_l: list = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            unique_h.append(h)
            unique_l.append(lab)
    if unique_l:
        ax.legend(unique_h, unique_l, loc="best", fontsize=8)


# ── Plot functions ────────────────────────────────────────────────────────────


def plot_xy_projection(
    mock: MockCatalog,
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
) -> None:
    """XY projection showing A/B points, true sphere outlines, and recovered centers.

    Matched recovered voids are marked with a green ``x``; unmatched voids
    (false positives or duplicates) are marked with a red ``+``.
    """
    bs = mock.A.box_size
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        mock.A.positions[:, 0], mock.A.positions[:, 1],
        s=1, c="steelblue", alpha=0.3, label="A (barriers)", rasterized=True,
    )
    ax.scatter(
        mock.B.positions[:, 0], mock.B.positions[:, 1],
        s=2, c="orange", alpha=0.5, label="B (tracers)", rasterized=True,
    )
    for c, r in zip(mock.true_void_centers, mock.true_void_radii):
        circle = plt.Circle((c[0], c[1]), r, fill=False, color="green", linewidth=1.5,
                             linestyle="--")
        ax.add_patch(circle)
    if voids:
        matched_set = set(summary.matched_void_indices.tolist())
        first_matched = min(matched_set) if matched_set else -1
        first_unmatched = next((i for i in range(len(voids)) if i not in matched_set), -1)
        for i, v in enumerate(voids):
            if i in matched_set:
                ax.scatter(v.center[0], v.center[1], marker="x", s=80, c="green",
                           zorder=5, label="matched" if i == first_matched else "")
            else:
                ax.scatter(v.center[0], v.center[1], marker="+", s=80, c="red",
                           zorder=5, label="unmatched/FP" if i == first_unmatched else "")
    ax.set_xlim(0, bs)
    ax.set_ylim(0, bs)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("XY projection")
    _legend_dedup(ax)
    _save_or_show(fig, outpath)


def plot_all_void_centers_overview(
    mock: MockCatalog,
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
) -> None:
    """XY overview of all true sphere outlines and all recovered void centers.

    True spheres are drawn as dashed circles, each labelled with their
    ``true_id`` at the true center.  Recovered void centers are coloured
    green (matched) or red (unmatched / false positive).  Unlike
    :func:`plot_xy_projection` this function omits the individual A/B point
    scatter so the void geometry is the focus.
    """
    bs = mock.A.box_size
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 300)

    matched_set = set(summary.matched_void_indices.tolist())

    # True spheres — dashed circles with true_id labels.
    for tid, (c, r) in enumerate(zip(mock.true_void_centers, mock.true_void_radii)):
        circle = plt.Circle((c[0], c[1]), r, fill=False, color="green",
                             linewidth=1.5, linestyle="--")
        ax.add_patch(circle)
        ax.text(c[0], c[1], str(tid), ha="center", va="center",
                fontsize=9, color="green", fontweight="bold")

    # Recovered void centers.
    first_m = next((i for i in range(len(voids)) if i in matched_set), -1)
    first_u = next((i for i in range(len(voids)) if i not in matched_set), -1)
    for i, v in enumerate(voids):
        if i in matched_set:
            ax.scatter(v.center[0], v.center[1], marker="x", s=80, c="green",
                       zorder=5, label="matched" if i == first_m else "")
        else:
            ax.scatter(v.center[0], v.center[1], marker="+", s=80, c="red",
                       zorder=5, label="unmatched/FP" if i == first_u else "")

    ax.set_xlim(0, bs)
    ax.set_ylim(0, bs)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    n_true = len(mock.true_void_radii)
    ax.set_title(
        f"All voids overview  |  {n_true} true,  {len(voids)} recovered"
        f"  ({summary.n_matched} matched,  {summary.n_missed} missed)"
    )
    _legend_dedup(ax)
    _save_or_show(fig, outpath)


def plot_slice_truth_vs_found(
    mock: MockCatalog,
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
    true_id: int = 0,
    axis: str = "z",
) -> None:
    """2D slab slice perpendicular to *axis* at the center of true sphere *true_id*.

    A and B points within a slab of half-thickness ``R_true / 4`` are shown.
    For the best-matched recovered void the following overlays are drawn:

    * **A boundary points** — purple scatter of A-catalog boundary particles in the slab.
    * **Alpha-shape slice** — solid purple line segments from the intersection of the
      alpha-shape surface with the cutting plane, computed via
      :func:`triangle_plane_intersections`.
    * **Recovered R_eff circle** — dashed purple circle centred on the recovered
      centroid (unwrapped around the true center to handle PBC correctly).
    * **Recovered center** — green star marker.

    The figure title includes the number of alpha-shape slice segments.
    """
    _AXIS_MAP = {"x": 0, "y": 1, "z": 2}
    if axis not in _AXIS_MAP:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    ax_idx = _AXIS_MAP[axis]
    other = [i for i in range(3) if i != ax_idx]
    ax_names = ["x", "y", "z"]

    center = mock.true_void_centers[true_id]
    radius = mock.true_void_radii[true_id]
    slab_half = max(radius / 4.0, 1.0)
    bs = mock.A.box_size

    def _in_slab(pos: np.ndarray) -> np.ndarray:
        dz = pos[:, ax_idx] - center[ax_idx]
        dz = dz - bs * np.rint(dz / bs)
        return np.abs(dz) < slab_half

    mask_A = _in_slab(mock.A.positions)
    mask_B = _in_slab(mock.B.positions)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        mock.A.positions[mask_A, other[0]], mock.A.positions[mask_A, other[1]],
        s=2, c="steelblue", alpha=0.4, label="A", rasterized=True,
    )
    ax.scatter(
        mock.B.positions[mask_B, other[0]], mock.B.positions[mask_B, other[1]],
        s=4, c="orange", alpha=0.6, label="B", rasterized=True,
    )
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(
        center[other[0]] + radius * np.cos(theta),
        center[other[1]] + radius * np.sin(theta),
        "g--", linewidth=1.5, label="true sphere",
    )

    n_alpha_segs = 0
    void, _ = select_best_match(voids, mock, true_id)
    if void is not None:
        # A boundary points in the slab.
        bdy = mock.A.positions[void.A_boundary_indices]
        mb = _in_slab(bdy)
        if mb.any():
            ax.scatter(
                bdy[mb, other[0]], bdy[mb, other[1]],
                s=12, c="purple", alpha=0.8, label="A boundary points", zorder=4,
            )

        # Alpha-shape slice: surface × cutting plane.
        if void.alpha_shape is not None and len(void.alpha_shape.accepted_tetrahedra) > 0:
            verts_uw = unwrap_points(mock.A.positions[void.A_boundary_indices], center, bs)
            faces = external_faces_from_tetrahedra(void.alpha_shape.accepted_tetrahedra)
            if len(faces) > 0:
                segs = triangle_plane_intersections(
                    verts_uw, faces, ax_idx, center[ax_idx]
                )
                n_alpha_segs = len(segs)
                for k, seg in enumerate(segs):
                    ax.plot(
                        [seg[0, other[0]], seg[1, other[0]]],
                        [seg[0, other[1]], seg[1, other[1]]],
                        color="purple", linewidth=1.2,
                        label="alpha-shape slice" if k == 0 else "",
                    )

        # Recovered R_eff circle centred on the recovered centroid unwrapped
        # around the true center so that PBC voids are visualised correctly.
        rc = unwrap_points(void.center[np.newaxis], center, bs)[0]
        ax.plot(
            rc[other[0]] + void.effective_radius * np.cos(theta),
            rc[other[1]] + void.effective_radius * np.sin(theta),
            color="purple", linestyle="--", linewidth=1.2,
            label=f"R_eff = {void.effective_radius:.1f}",
        )
        ax.scatter(
            rc[other[0]], rc[other[1]],
            marker="*", s=150, c="green", zorder=5, label="recovered center",
        )

    ax.set_xlim(0, bs)
    ax.set_ylim(0, bs)
    ax.set_aspect("equal")
    ax.set_xlabel(ax_names[other[0]])
    ax.set_ylabel(ax_names[other[1]])
    ax.set_title(
        f"Slice  true void {true_id}  |  R={radius:.1f}"
        f"  |  {axis}={center[ax_idx]:.1f} ±{slab_half:.1f}"
        f"  |  α-segments: {n_alpha_segs}"
    )
    _legend_dedup(ax)
    _save_or_show(fig, outpath)


def plot_3d_truth_and_recovered(
    mock: MockCatalog,
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
    true_id: int = 0,
) -> None:
    """3D scatter of A boundary points, recovered alpha-shape surface, and true sphere.

    All positions are unwrapped around the true sphere center so that voids
    straddling a periodic boundary are visualised correctly.  The accepted
    alpha-shape tetrahedra are rendered as a semi-transparent surface using
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    center = mock.true_void_centers[true_id]
    radius = mock.true_void_radii[true_id]
    bs = mock.A.box_size

    fig = plt.figure(figsize=(7, 6))
    ax3d = fig.add_subplot(111, projection="3d")

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    xs = center[0] + radius * np.outer(np.sin(phi), np.cos(theta))
    ys = center[1] + radius * np.outer(np.sin(phi), np.sin(theta))
    zs = center[2] + radius * np.outer(np.cos(phi), np.ones(len(theta)))
    ax3d.plot_wireframe(xs, ys, zs, alpha=0.15, color="green", linewidth=0.5,
                        label="true sphere")

    void, _ = select_best_match(voids, mock, true_id)
    if void is not None:
        # All A boundary positions unwrapped around the true center (handles PBC).
        bdy_pos = mock.A.positions[void.A_boundary_indices]
        verts_plot = unwrap_points(bdy_pos, center, bs)
        ax3d.scatter(*verts_plot.T, s=8, c="steelblue", alpha=0.5, label="A boundary")

        # Render the accepted alpha-shape tetrahedra as a surface mesh.
        # verts_plot[k] and void.alpha_shape.vertices[k] share the same ordering
        # (both derived from A_boundary_indices), so face connectivity transfers directly.
        if void.alpha_shape is not None and len(void.alpha_shape.accepted_tetrahedra) > 0:
            faces = external_faces_from_tetrahedra(void.alpha_shape.accepted_tetrahedra)
            if len(faces) > 0:
                triangles = verts_plot[faces]  # (M, 3, 3)
                mesh = Poly3DCollection(
                    triangles, alpha=0.25, facecolor="steelblue", edgecolor="none",
                )
                ax3d.add_collection3d(mesh)

        # Unwrap the recovered center around the true center for PBC correctness.
        rc = unwrap_points(void.center[np.newaxis], center, bs)[0]
        ax3d.scatter(rc[0], rc[1], rc[2], s=80, c="green", marker="*", zorder=5,
                     label="recovered center")

    ax3d.scatter(center[0], center[1], center[2], s=80, c="red", marker="^", zorder=5,
                 label="true center")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    cx, cy, cz = center
    ax3d.set_title(
        f"True void {true_id}  |  R={radius:.1f}  |  "
        f"c=({cx:.1f}, {cy:.1f}, {cz:.1f})"
    )
    ax3d.legend(loc="upper left", fontsize=8)
    _save_or_show(fig, outpath)


def plot_radial_profile(
    mock: MockCatalog,
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
    true_id: int = 0,
    normalize_by_mean: bool = False,
) -> None:
    """Radial number-density profiles of A and B around the true void center.

    Vertical dashed lines mark the true sphere radius and (if matched) the
    recovered effective radius.

    Parameters
    ----------
    normalize_by_mean:
        When ``True``, each profile is divided by the mean catalog number density
        so that unity corresponds to the background level.
    """
    center = mock.true_void_centers[true_id]
    radius = mock.true_void_radii[true_id]
    bs = mock.A.box_size

    r_max = min(2.5 * radius, bs / 2.0)
    r_bins = np.linspace(0.0, r_max, 40)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    rho_A = radial_profile(mock.A.positions, center, bs, r_bins,
                           normalize_by_mean=normalize_by_mean)
    rho_B = radial_profile(mock.B.positions, center, bs, r_bins,
                           normalize_by_mean=normalize_by_mean)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_mid, rho_A, "b-o", markersize=3, label="A density")
    ax.plot(r_mid, rho_B, "r-s", markersize=3, label="B density")
    ax.axvline(radius, color="green", linestyle="--", linewidth=1.5,
               label=f"R_true = {radius:.1f}")

    void, _ = select_best_match(voids, mock, true_id)
    if void is not None:
        ax.axvline(void.effective_radius, color="purple", linestyle=":", linewidth=1.5,
                   label=f"R_rec = {void.effective_radius:.1f}")

    ax.set_xlabel("r")
    ylabel = "n / mean_n" if normalize_by_mean else "n / V_shell"
    ax.set_ylabel(ylabel)
    cx, cy, cz = center
    ax.set_title(
        f"Radial profiles  |  true void {true_id}  |  R={radius:.1f}"
        f"  |  c=({cx:.1f}, {cy:.1f}, {cz:.1f})"
    )
    ax.legend(loc="best", fontsize=8)
    _save_or_show(fig, outpath)


def plot_component_size_distribution(
    run: FinderRun,
    outpath: str | Path | None = None,
) -> None:
    """Histogram of B component sizes (number of B sub-points per component)."""
    labels = run.component_labels
    _, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    n_bins = max(10, len(counts) // 5 + 1)
    ax.hist(counts, bins=n_bins, color="steelblue", edgecolor="white")
    ax.set_xlabel("Component size (# B points)")
    ax.set_ylabel("Count")
    ax.set_title("B component size distribution")
    if counts.max() > 1:
        ax.set_yscale("log")
    _save_or_show(fig, outpath)


def plot_boundary_size_distribution(
    run: FinderRun,
    outpath: str | Path | None = None,
) -> None:
    """Overlapping step histograms of A boundary sizes at all four pipeline stages.

    The four stages—veto, shell, selected, and final (after dilation)—are shown
    in the same panel so their distributions can be compared directly.
    """
    stages = [
        ("veto",     run.veto_boundary_sets,     "steelblue"),
        ("shell",    run.shell_boundary_sets,     "orange"),
        ("selected", run.selected_boundary_sets,  "green"),
        ("final",    run.final_boundary_sets,     "coral"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for label, bsets, color in stages:
        sizes = [len(v) for v in bsets.values()]
        if not sizes:
            continue
        any_data = True
        all_sizes = np.asarray(sizes)
        if all_sizes.max() <= 0:
            continue
        bins = np.arange(0, all_sizes.max() + 2)
        ax.hist(all_sizes, bins=bins, histtype="step", color=color, linewidth=1.5,
                label=f"{label} (n={len(sizes)})")

    if not any_data:
        ax.text(0.5, 0.5, "No boundaries", ha="center", va="center",
                transform=ax.transAxes)
        _save_or_show(fig, outpath)
        return

    ax.set_xlabel("Boundary size (# A points)")
    ax.set_ylabel("Count")
    ax.set_title("A boundary size distribution by pipeline stage")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    _save_or_show(fig, outpath)


def plot_alpha_diagnostics(
    voids: list[Void],
    summary: ValidationSummary,
    outpath: str | Path | None = None,
) -> None:
    """Scatter of accepted vs total tetrahedra, and alpha fraction per void.

    Green markers indicate matched voids; salmon markers indicate unmatched.
    """
    alpha_void_indices = [i for i, v in enumerate(voids) if v.alpha_shape is not None]
    alpha_voids = [voids[i] for i in alpha_void_indices]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if not alpha_voids:
        for ax in axes:
            ax.text(0.5, 0.5, "No alpha shape data", ha="center", va="center",
                    transform=ax.transAxes)
        _save_or_show(fig, outpath)
        return

    n_total = np.array([v.alpha_shape.n_tetrahedra_total for v in alpha_voids])
    n_acc = np.array([v.alpha_shape.n_tetrahedra_accepted for v in alpha_voids])
    fracs = np.array([v.alpha_shape.alpha_fraction for v in alpha_voids])

    matched_set = set(summary.matched_void_indices.tolist())
    colors = ["green" if orig_i in matched_set else "salmon"
              for orig_i in alpha_void_indices]

    ax0, ax1 = axes
    ax0.scatter(n_total, n_acc, c=colors, s=30, edgecolors="gray", linewidths=0.3)
    lim = max(int(n_total.max()), int(n_acc.max())) * 1.05
    ax0.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax0.set_xlabel("n_tetrahedra_total")
    ax0.set_ylabel("n_tetrahedra_accepted")
    ax0.set_title("Alpha shape: accepted vs total")
    ax0.legend(fontsize=8)

    ax1.scatter(range(len(alpha_voids)), fracs, c=colors, s=30,
                edgecolors="gray", linewidths=0.3)
    ax1.set_xlabel("Void index")
    ax1.set_ylabel("alpha_fraction")
    ax1.set_title("Alpha fraction per void")
    ax1.set_ylim(0, 1)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="green", label="matched"),
        Patch(color="salmon", label="unmatched"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, outpath)
