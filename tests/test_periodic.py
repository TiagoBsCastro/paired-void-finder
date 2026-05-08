import numpy as np

from paired_void_finder.periodic import (
    distance_point_to_segment_periodic,
    periodic_delta,
    periodic_distance,
    unwrap_points,
    wrap_points,
)


def test_periodic_delta_across_boundary():
    L = 10.0
    d = periodic_delta(np.array([0.2, 0.0, 0.0]), np.array([9.8, 0.0, 0.0]), L)
    assert np.allclose(d, np.array([0.4, 0.0, 0.0]))


def test_periodic_distance_across_boundary():
    L = 10.0
    d = periodic_distance(np.array([0.2, 0.0, 0.0]), np.array([9.8, 0.0, 0.0]), L)
    assert np.isclose(d, 0.4)


def test_wrap_and_unwrap():
    L = 10.0
    pts = np.array([[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]])
    ref = np.array([9.9, 5.0, 5.0])
    unwrapped = unwrap_points(pts, ref, L)
    assert np.all(np.abs(unwrapped[:, 0] - ref[0]) < 0.5)
    assert np.all((wrap_points(unwrapped, L) >= 0.0) & (wrap_points(unwrapped, L) < L))


def test_point_to_segment_periodic():
    L = 10.0
    p = np.array([0.0, 1.0, 0.0])
    a = np.array([9.5, 0.0, 0.0])
    b = np.array([0.5, 0.0, 0.0])
    d = distance_point_to_segment_periodic(p, a, b, L)
    assert np.isclose(d, 1.0)
