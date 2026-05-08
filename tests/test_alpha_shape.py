import numpy as np

from paired_void_finder.alpha_shape import alpha_shape_3d, tetra_volume


def test_tetra_volume_unit_simplex():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert np.isclose(tetra_volume(points), 1.0 / 6.0)


def test_alpha_shape_from_cube_corners_has_positive_volume():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    res = alpha_shape_3d(points, box_size=10.0, R_alpha=10.0, reference=np.array([0.5, 0.5, 0.5]))
    assert res.volume > 0.0
    assert res.effective_radius > 0.0
    assert res.n_tetrahedra_total > 0
    assert res.n_tetrahedra_accepted > 0
    assert res.n_tetrahedra_accepted <= res.n_tetrahedra_total
    assert 0.0 < res.alpha_fraction <= 1.0
