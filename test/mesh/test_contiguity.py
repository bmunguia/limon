"""Writers read array buffers as C-contiguous; views with other strides must not be scrambled."""

import numpy as np
import pytest
from limon.mesh import load_mesh, load_solution, write_mesh, write_solution


@pytest.mark.parametrize('suffix', ['.solb', '.dat', '.csv'])
def test_non_contiguous_solution_round_trips(tmp_path, suffix):
    n = 50
    rng = np.random.default_rng(0)
    fortran = np.asfortranarray(rng.random((n, 3)))  # symmetric tensor, F-ordered
    sliced = rng.random((n, 6))[:, ::2]  # strided view
    scalar = rng.random(2 * n)[::2]
    assert not fortran.flags['C_CONTIGUOUS'] and not sliced.flags['C_CONTIGUOUS']
    solution = {'Metric': fortran, 'Vector': sliced[:, :2], 'Scalar': scalar}

    path = tmp_path / f'sol{suffix}'
    write_solution(path, solution, n, 2)
    back = load_solution(path, n, 2)['solution']

    for (name, field), got in zip(solution.items(), back.values()):
        np.testing.assert_allclose(got, field, rtol=1e-6, err_msg=name)


@pytest.mark.parametrize('suffix', ['.meshb', '.su2'])
def test_non_contiguous_mesh_arrays_round_trip(tmp_path, suffix):
    mesh = load_mesh('data/square/square.su2')
    scrambled = dict(
        mesh,
        coords=np.asfortranarray(mesh['coords']),
        elements={k: np.asfortranarray(v) for k, v in mesh['elements'].items()},
        boundaries={k: np.asfortranarray(v) for k, v in mesh['boundaries'].items()},
    )
    path = tmp_path / f'square{suffix}'
    write_mesh(path, scrambled)
    back = load_mesh(path)

    np.testing.assert_allclose(back['coords'], mesh['coords'])
    for key, value in mesh['elements'].items():
        np.testing.assert_array_equal(back['elements'][key][:, :-1], value[:, :-1])
