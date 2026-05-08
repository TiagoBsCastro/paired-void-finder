import numpy as np

from paired_void_finder.catalogs import Catalog
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.periodic import periodic_distance


def test_catalog_wraps_positions_into_box():
    """Catalog.__post_init__ wraps positions outside [0, box_size) at construction."""
    raw = np.array([[-1.0, 101.0, 50.0], [100.0, -0.1, 50.0]])
    cat = Catalog(raw, None, box_size=100.0, name="test")
    assert np.all(cat.positions >= 0.0), "positions contain negative values after wrapping"
    assert np.all(cat.positions < 100.0), "positions >= box_size after wrapping"
    np.testing.assert_allclose(cat.positions[0, 0], 99.0)   # -1 % 100
    np.testing.assert_allclose(cat.positions[0, 1], 1.0)    # 101 % 100
    np.testing.assert_allclose(cat.positions[1, 0], 0.0)    # 100 % 100
    np.testing.assert_allclose(cat.positions[1, 1], 99.9)   # -0.1 % 100


def test_geometry_mock_splits_inside_and_outside():
    mock = make_swiss_cheese_mock(
        box_size=50.0,
        n_points=4000,
        void_centers=np.array([[25.0, 25.0, 25.0]]),
        void_radii=np.array([8.0]),
        mode="geometry",
        seed=1,
    )
    assert len(mock.A.positions) > 0
    assert len(mock.B.positions) > 0
    dA = periodic_distance(mock.A.positions, mock.true_void_centers[0], mock.A.box_size)
    dB = periodic_distance(mock.B.positions, mock.true_void_centers[0], mock.B.box_size)
    assert np.all(dA > mock.true_void_radii[0])
    assert np.all(dB <= mock.true_void_radii[0])


def test_veto_mock_adds_exterior_b_decoys():
    mock = make_swiss_cheese_mock(
        box_size=50.0,
        n_points=4000,
        void_centers=np.array([[25.0, 25.0, 25.0]]),
        void_radii=np.array([8.0]),
        mode="veto",
        exterior_b_fraction=0.1,
        seed=1,
    )
    dB = periodic_distance(mock.B.positions, mock.true_void_centers[0], mock.B.box_size)
    assert np.any(dB > mock.true_void_radii[0])
