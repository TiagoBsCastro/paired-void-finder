import numpy as np

from paired_void_finder.catalogs import Edge
from paired_void_finder.veto import apply_a_barrier_veto


def test_a_point_vetoes_segment():
    L = 10.0
    B = np.array([[2.0, 5.0, 5.0], [6.0, 5.0, 5.0]])
    A = np.array([[4.0, 5.1, 5.0]])
    radii = np.array([0.2])
    out = apply_a_barrier_veto(B, A, [Edge(0, 1)], radii, L)
    assert len(out) == 1
    assert not out[0].accepted
    assert out[0].veto_halos == (0,)
