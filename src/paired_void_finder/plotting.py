"""Basic plotting utilities for diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .catalogs import MockCatalog, Void


def plot_xy_projection(mock: MockCatalog, voids: list[Void], outpath: str | None = None) -> None:
    """Plot an xy projection of A, B, true voids, and recovered centers."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mock.A.positions[:, 0], mock.A.positions[:, 1], s=2, label="A")
    ax.scatter(mock.B.positions[:, 0], mock.B.positions[:, 1], s=2, label="B")
    for c, r in zip(mock.true_void_centers, mock.true_void_radii):
        circle = plt.Circle((c[0], c[1]), r, fill=False, linewidth=1.5)
        ax.add_patch(circle)
    if voids:
        centers = np.asarray([v.center for v in voids])
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=40, label="recovered")
    ax.set_aspect("equal")
    ax.set_xlim(0, mock.A.box_size)
    ax.set_ylim(0, mock.A.box_size)
    ax.legend(loc="best")
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
