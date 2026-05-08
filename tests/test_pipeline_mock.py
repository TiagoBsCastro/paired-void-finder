import numpy as np

from paired_void_finder.catalogs import FinderParameters
from paired_void_finder.mocks import make_swiss_cheese_mock
from paired_void_finder.voids import run_void_finder


def test_pipeline_imports_and_runs_without_crashing():
    mock = make_swiss_cheese_mock(
        box_size=50.0,
        n_points=512,
        void_centers=np.array([[25.0, 25.0, 25.0]]),
        void_radii=np.array([10.0]),
        mode="veto",
        exterior_b_fraction=0.05,
        seed=2,
    )
    params = FinderParameters(
        b_BB=1.6,
        eta=0.5,
        N_veto=5,
        b_grow=0.7,
        lambda_alpha=3.0,
        N_B_min=2,
        N_A_min=4,
    )
    voids = run_void_finder(mock.A, mock.B, params)
    assert isinstance(voids, list)
