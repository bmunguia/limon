from pathlib import Path

import numpy as np
import pytest
from limon.geometry import surface_metric
from limon.mesh import load_mesh, load_mesh_and_solution, write_mesh_and_solution

CYLINDER = Path('data/cylinder/cylinder_20m_quasi_structured.su2')
GOLDEN = Path('data/cylinder/metric_geo_golden.csv')
GEODEV = {'cylinder': 2.0, 'farfield': 10.0}


@pytest.fixture(scope='module')
def mesh():
    return load_mesh(CYLINDER)


def _eigenvalues(metric):
    xx, xy, yy = metric.T
    full = np.stack([np.stack([xx, xy], axis=-1), np.stack([xy, yy], axis=-1)], axis=-2)
    eig = np.linalg.eigvalsh(full)
    return eig[:, 0], eig[:, 1]


def test_matches_su2_golden(mesh):
    """SU2's Metric_Geo for the cylinder mesh, ANGLE mode, SU2's default size bounds."""
    golden = np.loadtxt(GOLDEN, delimiter=',')
    nodes = golden[:, 0].astype(int)

    metric = surface_metric(mesh, GEODEV, mode='angle', hmin=1e-8, hmax=1e8)

    assert metric.shape == (mesh['num_point'], 3)
    np.testing.assert_allclose(metric[nodes], golden[:, 1:], rtol=1e-8, atol=1e-9 * golden[:, 1:].max())
    interior = np.setdiff1d(np.arange(mesh['num_point']), nodes)
    np.testing.assert_allclose(metric[interior], np.tile([1e-16, 0.0, 1e-16], (len(interior), 1)))


def test_bounds_give_a_valid_metric(mesh):
    metric = surface_metric(mesh, GEODEV, hmin=1e-6, hmax=25.0)
    lo, hi = _eigenvalues(metric)
    assert lo.min() >= 1 / 25.0**2 - 1e-12
    assert hi.max() <= 1 / 1e-6**2
    assert metric.flags['C_CONTIGUOUS'] and metric.dtype == np.float64


def test_hausdorff_mode_uses_sagitta(mesh):
    # circle of radius 0.5: kappa = 2, h = sqrt(8 * deviation / kappa)
    metric = surface_metric(mesh, {'cylinder': 0.01}, mode='hausdorff', hmin=1e-6, hmax=25.0)
    _, hi = _eigenvalues(metric)
    np.testing.assert_allclose(hi.max(), 1.0 / (8 * 0.01 / 2.0), rtol=1e-3)


def test_non_positive_deviation_gives_hmax(mesh):
    metric = surface_metric(mesh, {'cylinder': 0.0, 'farfield': -1.0}, hmin=1e-6, hmax=25.0)
    np.testing.assert_allclose(metric, np.tile([1 / 625.0, 0.0, 1 / 625.0], (mesh['num_point'], 1)))


def test_corner_nodes_stay_isotropic(mesh):
    corner = int(mesh['boundaries']['Edges'][mesh['boundaries']['Edges'][:, -1] == 1][0, 0])
    with_corner = dict(mesh, boundaries=dict(mesh['boundaries'], Corners=np.array([corner], dtype=np.uint32)))
    metric = surface_metric(with_corner, GEODEV, hmin=1e-6, hmax=25.0)
    np.testing.assert_allclose(metric[corner], [1 / 625.0, 0.0, 1 / 625.0])
    assert surface_metric(mesh, GEODEV, hmin=1e-6, hmax=25.0)[corner, 0] != pytest.approx(1 / 625.0)


def test_first_marker_wins_on_shared_nodes(mesh):
    edges = np.array(mesh['boundaries']['Edges'])
    shared = dict(mesh, boundaries=dict(mesh['boundaries'], Edges=np.vstack([edges, edges[edges[:, -1] == 1] * [1, 1, 0] + [0, 0, 2]]).astype(np.uint32)))
    first = surface_metric(shared, {'cylinder': 2.0, 'farfield': 10.0}, hmin=1e-6, hmax=25.0)
    swapped = surface_metric(shared, {'farfield': 10.0, 'cylinder': 2.0}, hmin=1e-6, hmax=25.0)
    node = int(edges[edges[:, -1] == 1][0, 0])
    assert first[node, 2] != pytest.approx(swapped[node, 2])


def test_unknown_marker_raises(mesh):
    with pytest.raises(ValueError, match='nope'):
        surface_metric(mesh, {'nope': 1.0})


@pytest.mark.parametrize('kwargs', [{'mode': 'degrees'}, {'hmin': 1.0, 'hmax': 0.5}, {'hmin': 0.0}])
def test_invalid_arguments_raise(mesh, kwargs):
    with pytest.raises(ValueError):
        surface_metric(mesh, GEODEV, **kwargs)


def test_3d_is_not_implemented():
    with pytest.raises(NotImplementedError, match='phd-greenlight'):
        surface_metric({'dim': 3}, GEODEV)


def test_survives_a_gmf_round_trip(mesh, tmp_path):
    metric = surface_metric(mesh, GEODEV, hmin=1e-6, hmax=25.0)
    write_mesh_and_solution(tmp_path / 'back.meshb', tmp_path / 'back.solb', dict(mesh, solution={'Metric': metric}))
    back = load_mesh_and_solution(tmp_path / 'back.meshb', tmp_path / 'back.solb')
    np.testing.assert_allclose(back['solution']['Metric'], metric)
    assert back['markers'] == mesh['markers']
