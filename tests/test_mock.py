import numpy as np

from paired_void_finder.catalogs import Catalog
from paired_void_finder.mocks import generate_random_void_spheres, make_swiss_cheese_mock
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


def test_catalog_position_wrapping_pipeline_safe():
    """Positions outside [0, box_size) are silently wrapped, and the pipeline runs without error."""
    from paired_void_finder.catalogs import FinderParameters
    from paired_void_finder.voids import run_void_finder

    rng = np.random.default_rng(7)
    box_size = 50.0
    n = 300

    # Generate positions slightly outside the box (in range [-5, 55]).
    raw_pos = rng.uniform(-5.0, 55.0, (n, 3))
    # Intentional: some positions will be in (-5, 0) or (50, 55).
    A_pos = raw_pos[:200]
    B_pos = raw_pos[200:]

    cat_A = Catalog(A_pos, None, box_size=box_size, name="A")
    cat_B = Catalog(B_pos, None, box_size=box_size, name="B")

    # Verify wrapping happened at construction.
    assert np.all(cat_A.positions >= 0.0) and np.all(cat_A.positions < box_size)
    assert np.all(cat_B.positions >= 0.0) and np.all(cat_B.positions < box_size)

    # The pipeline must not raise (e.g., no cKDTree out-of-bounds exception).
    params = FinderParameters(boundary_mode="shell", enable_veto=False)
    voids = run_void_finder(cat_A, cat_B, params)
    assert isinstance(voids, list)


# ── generate_random_void_spheres ──────────────────────────────────────────────


def test_generate_random_void_spheres_returns_correct_shapes():
    """generate_random_void_spheres returns (centers, radii) with expected shapes."""
    centers, radii = generate_random_void_spheres(
        n_holes=5,
        box_size=100.0,
        radii=8.0,
        seed=0,
        min_separation_factor=2.0,
    )
    assert centers.shape == (5, 3), f"Expected centers shape (5, 3), got {centers.shape}"
    assert radii.shape == (5,), f"Expected radii shape (5,), got {radii.shape}"
    np.testing.assert_array_equal(radii, 8.0)
    # All centres must be inside the box.
    assert np.all(centers >= 0.0) and np.all(centers < 100.0)


def test_generate_random_void_spheres_satisfies_non_overlap():
    """All placed holes satisfy the periodic non-overlap condition."""
    n_holes = 4
    box_size = 150.0
    radius = 10.0
    min_sep = 2.5
    centers, radii = generate_random_void_spheres(
        n_holes=n_holes,
        box_size=box_size,
        radii=radius,
        seed=99,
        min_separation_factor=min_sep,
    )
    for i in range(n_holes):
        for j in range(i + 1, n_holes):
            d = float(
                periodic_distance(centers[i : i + 1], centers[j], box_size)[0]
            )
            min_required = min_sep * (radii[i] + radii[j])
            assert d >= min_required, (
                f"Holes {i} and {j} too close: d={d:.3f} < min={min_required:.3f}"
            )
