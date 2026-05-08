import numpy as np

from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.periodic import periodic_distance


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
