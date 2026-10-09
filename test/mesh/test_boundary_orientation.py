"""Written meshes follow SU2's interior and boundary orientation checks."""

import numpy as np
import pytest
from limon.mesh import load_mesh, write_mesh
from limon.mesh.orientation import orient_mesh


def _mesh(coords, elements, boundaries):
    return {
        'coords': np.array(coords, dtype=float),
        'elements': {k: np.array(v, dtype=np.uint32) for k, v in elements.items()},
        'boundaries': {k: np.array(v, dtype=np.uint32) for k, v in boundaries.items()},
        'dim': len(coords[0]),
        'num_point': len(coords),
        'markers': {1: 'wall'},
    }


SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _rows(array):
    return sorted(map(tuple, array[:, :-1].tolist()))


def test_2d_interior_and_boundary_follow_su2_rules(tmp_path):
    # Clockwise triangles; boundary edges given in both directions
    mesh = _mesh(SQUARE, {'Triangles': [[0, 2, 1, 0], [0, 3, 2, 0]]}, {'Edges': [[1, 0, 1], [1, 2, 1], [2, 3, 1], [0, 3, 1]]})
    fixed = orient_mesh(mesh)

    tri = fixed['elements']['Triangles'][:, :3].astype(int)
    for a, b, c in tri:
        p = np.array(SQUARE, dtype=float)
        assert _cross2(p[b] - p[a], p[c] - p[a]) > 0  # Counter-clockwise
    # The domain lies to the left of every edge
    for a, b, _ in fixed['boundaries']['Edges'].astype(int):
        p = np.array(SQUARE, dtype=float)
        centre = np.array([0.5, 0.5])
        assert _cross2(p[b] - p[a], centre - p[a]) > 0

    path = tmp_path / 'square.su2'
    write_mesh(path, mesh)
    back = load_mesh(path)
    assert _rows(back['boundaries']['Edges']) == _rows(fixed['boundaries']['Edges'])
    assert _rows(back['elements']['Triangles']) == _rows(fixed['elements']['Triangles'])


def test_inputs_are_not_modified():
    mesh = _mesh(SQUARE, {'Triangles': [[0, 2, 1, 0], [0, 3, 2, 0]]}, {'Edges': [[1, 0, 1], [1, 2, 1]]})
    before = (mesh['elements']['Triangles'].copy(), mesh['boundaries']['Edges'].copy())
    orient_mesh(mesh)
    np.testing.assert_array_equal(mesh['elements']['Triangles'], before[0])
    np.testing.assert_array_equal(mesh['boundaries']['Edges'], before[1])


def test_consistent_mesh_is_unchanged():
    mesh = _mesh(SQUARE, {'Triangles': [[0, 1, 2, 0], [0, 2, 3, 0]]}, {'Edges': [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1]]})
    fixed = orient_mesh(mesh)
    np.testing.assert_array_equal(fixed['elements']['Triangles'], mesh['elements']['Triangles'])
    np.testing.assert_array_equal(fixed['boundaries']['Edges'], mesh['boundaries']['Edges'])


def test_2d_quadrilateral_is_flipped_when_both_triangles_fail():
    mesh = _mesh(SQUARE, {'Quadrilaterals': [[0, 3, 2, 1, 0]]}, {})
    np.testing.assert_array_equal(orient_mesh(mesh)['elements']['Quadrilaterals'], [[0, 1, 2, 3, 0]])


def test_2d_gmf_output_is_oriented(tmp_path):
    mesh = _mesh(SQUARE, {'Triangles': [[0, 2, 1, 0], [0, 3, 2, 0]]}, {'Edges': [[1, 0, 1], [2, 1, 1]]})
    path = tmp_path / 'square.meshb'
    write_mesh(path, mesh)
    back = load_mesh(path)
    assert _rows(back['elements']['Triangles']) == _rows(orient_mesh(mesh)['elements']['Triangles'])
    assert _rows(back['boundaries']['Edges']) == _rows(orient_mesh(mesh)['boundaries']['Edges'])


@pytest.mark.parametrize(
    'name, nodes, expected, mirror',
    [
        ('Tetrahedra', [0, 1, 2, 4], [1, 0, 2, 4], True),
        ('Hexahedra', [0, 1, 2, 3, 4, 5, 6, 7], [0, 3, 2, 1, 4, 7, 6, 5], True),
        ('Pyramids', [0, 1, 2, 3, 4], [0, 3, 2, 1, 4], True),
        # SU2 prisms have the base normal pointing away from the top face
        ('Prisms', [0, 1, 2, 4, 5, 6], [1, 0, 2, 5, 4, 6], False),
    ],
)
def test_3d_elements_are_flipped_like_su2(name, nodes, expected, mirror):
    coords = np.array(CUBE, dtype=float)
    if mirror:
        coords[:, 2] *= -1.0
    mesh = _mesh(coords.tolist(), {name: [nodes + [0]]}, {})
    np.testing.assert_array_equal(orient_mesh(mesh)['elements'][name][0, :-1], expected)


def test_3d_boundary_triangle_points_into_the_domain():
    tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mesh = _mesh(tet, {'Tetrahedra': [[0, 1, 2, 3, 0]]}, {'Triangles': [[0, 1, 2, 1], [0, 2, 3, 1]]})
    fixed = orient_mesh(mesh)
    for a, b, c, _ in fixed['boundaries']['Triangles'].astype(int):
        p = np.array(tet, dtype=float)
        d = (set(range(4)) - {a, b, c}).pop()
        assert np.dot(np.cross(p[b] - p[a], p[c] - p[a]), p[d] - p[a]) > 0


def test_3d_gmf_output_is_left_as_given(tmp_path):
    tet = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]  # Negative volume
    mesh = _mesh(tet, {'Tetrahedra': [[0, 1, 2, 3, 0]]}, {'Triangles': [[0, 1, 2, 1]]})
    path = tmp_path / 'tet.meshb'
    write_mesh(path, mesh)
    np.testing.assert_array_equal(load_mesh(path)['elements']['Tetrahedra'][:, :4], [[0, 1, 2, 3]])
