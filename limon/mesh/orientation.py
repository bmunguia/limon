import numpy as np

# Node counts of the interior element types, keyed as in ``mesh_data['elements']``.
_NUM_NODE = {'Triangles': 3, 'Quadrilaterals': 4, 'Tetrahedra': 4, 'Pyramids': 5, 'Prisms': 6, 'Hexahedra': 8}
_ELEMENTS_2D = ('Triangles', 'Quadrilaterals')
_ELEMENTS_3D = ('Tetrahedra', 'Pyramids', 'Prisms', 'Hexahedra')

# Sub-tetrahedra (as node-index quadruples) whose volumes must be positive, and the node swaps that flip an element.
# Both follow SU2's CPhysicalGeometry::Check_IntElem_Orientation and the primal grid Change_Orientation methods.
_TETS = {
    'Tetrahedra': [(0, 1, 2, 3)],
    'Pyramids': [(0, 1, 2, 4), (2, 3, 0, 4)],
    'Prisms': [(0, 2, 1, 3), (3, 4, 5, 2)],
    'Hexahedra': [(0, 1, 2, 5), (0, 2, 3, 7), (4, 6, 5, 1), (4, 7, 6, 3)],
}
_SWAPS = {
    'Triangles': [(0, 2)],
    'Quadrilaterals': [(1, 3)],
    'Tetrahedra': [(0, 1)],
    'Pyramids': [(1, 3)],
    'Prisms': [(0, 1), (3, 4)],
    'Hexahedra': [(1, 3), (5, 7)],
}


def _signed_area(coords: np.ndarray, nodes: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """z-component of (p_j - p_i) x (p_k - p_i) for each row of ``nodes``."""
    p0, p1, p2 = coords[nodes[:, i], :2], coords[nodes[:, j], :2], coords[nodes[:, k], :2]
    a, b = p1 - p0, p2 - p0
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _tet_volume(coords: np.ndarray, nodes: np.ndarray, i: int, j: int, k: int, l: int) -> np.ndarray:
    """((p_j - p_i) x (p_k - p_i)) . (p_l - p_i) for each row of ``nodes``."""
    p0 = coords[nodes[:, i]]
    return np.einsum(
        'ij,ij->i',
        np.cross(coords[nodes[:, j]] - p0, coords[nodes[:, k]] - p0),
        coords[nodes[:, l]] - p0,
    )


def _swap(array: np.ndarray, rows: np.ndarray, swaps: list[tuple[int, int]]) -> None:
    for i, j in swaps:
        array[rows, i], array[rows, j] = array[rows, j].copy(), array[rows, i].copy()


def _flip_mask(coords: np.ndarray, name: str, nodes: np.ndarray) -> np.ndarray:
    if name == 'Triangles':
        return _signed_area(coords, nodes, 0, 1, 2) < 0.0
    if name == 'Quadrilaterals':
        return (_signed_area(coords, nodes, 0, 1, 2) < 0.0) & (_signed_area(coords, nodes, 0, 2, 3) < 0.0)
    return np.logical_and.reduce([_tet_volume(coords, nodes, *tet) < 0.0 for tet in _TETS[name]])


def orient_elements(coords: np.ndarray, elements: dict) -> dict:
    r"""Orient interior elements as SU2's ``Check_IntElem_Orientation`` does.

    Triangles must have a positive signed area (counter-clockwise), quadrilaterals are checked as two
    triangles, and 3-D elements as tetrahedra with positive volume. An element is flipped with the same node
    swap as SU2 only when every test fails; elements that pass some tests and fail others are left alone.

    Args:
        coords: Node coordinates, shape (num_point, dim).
        elements: Dictionary mapping element type to arrays of node indices plus a trailing reference column.

    Returns:
        A new elements dictionary. The input arrays are not modified.
    """
    dim = coords.shape[1]
    allowed = _ELEMENTS_2D if dim == 2 else _ELEMENTS_3D
    out = dict(elements)
    for name in allowed:
        array = elements.get(name)
        if array is None or len(array) == 0:
            continue
        nodes = np.asarray(array[:, :_NUM_NODE[name]], dtype=np.int64)
        flip = np.flatnonzero(_flip_mask(coords, name, nodes))
        if len(flip) == 0:
            continue
        oriented = np.array(array, order='C', copy=True)
        _swap(oriented, flip, _SWAPS[name])
        out[name] = oriented
    return out


class _Incidence:
    """Node-to-element incidence over all interior elements, in CSR form."""

    def __init__(self, elements: dict, num_point: int):
        self.elements = elements
        self.names = [name for name in _NUM_NODE if elements.get(name) is not None and len(elements[name])]
        sizes = [len(elements[name]) for name in self.names]
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        nodes = [np.asarray(elements[name][:, :_NUM_NODE[name]], dtype=np.int64) for name in self.names]
        flat_nodes = np.concatenate([n.ravel() for n in nodes]) if nodes else np.empty(0, dtype=np.int64)
        flat_ids = (
            np.concatenate([np.repeat(np.arange(start, start + len(n)), n.shape[1])
                            for start, n in zip(self.starts, nodes)])
            if nodes else np.empty(0, dtype=np.int64)
        )
        order = np.argsort(flat_nodes, kind='stable')
        self.ids = flat_ids[order]
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(flat_nodes, minlength=num_point))])

    def domain_point(self, face: np.ndarray) -> int | None:
        """First node, in element order, of an element containing ``face`` that is not on the face."""
        for element_id in np.unique(self.ids[self.offsets[face[0]]:self.offsets[face[0] + 1]]):
            kind = np.searchsorted(self.starts, element_id, side='right') - 1
            name = self.names[kind]
            nodes = self.elements[name][element_id - self.starts[kind], :_NUM_NODE[name]]
            if np.isin(face, nodes).all():
                for node in nodes:
                    if node not in face:
                        return int(node)
        return None


def orient_boundaries(coords: np.ndarray, elements: dict, boundaries: dict) -> dict:
    r"""Orient boundary elements as SU2's ``Check_BoundElem_Orientation`` does.

    The reference is a node of an adjacent interior element that is not on the boundary element. 2-D edges
    must have that node to their left, triangular faces must form a positive tetrahedron with it, and
    quadrilateral faces are flipped when at least 3 of their 4 sub-triangle tests fail. Elements without an
    adjacent interior element are left unchanged.

    Args:
        coords: Node coordinates, shape (num_point, dim).
        elements: Interior elements, already oriented.
        boundaries: Dictionary mapping boundary element type to arrays of node indices plus a reference column.

    Returns:
        A new boundaries dictionary. The input arrays are not modified.
    """
    dim = coords.shape[1]
    incidence = _Incidence(elements, len(coords))
    out = dict(boundaries)

    def lookup(array, num_node):
        nodes = np.asarray(array[:, :num_node], dtype=np.int64)
        refs = np.full(len(nodes), -1, dtype=np.int64)
        for row, face in enumerate(nodes):
            point = incidence.domain_point(face)
            if point is not None:
                refs[row] = point
        return nodes, refs

    def finish(name, array, flip):
        if flip.any():
            oriented = np.array(array, order='C', copy=True)
            _swap(oriented, np.flatnonzero(flip), _SWAPS_BOUNDARY[name])
            out[name] = oriented

    array = boundaries.get('Edges')
    if dim == 2 and array is not None and len(array):
        nodes, point = lookup(array, 2)
        found = point >= 0
        p0, p1, pd = coords[nodes[:, 0], :2], coords[nodes[:, 1], :2], coords[np.where(found, point, 0), :2]
        a, b = p1 - p0, pd - p0
        finish('Edges', array, found & (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0.0))

    if dim == 3:
        array = boundaries.get('Triangles')
        if array is not None and len(array):
            nodes, point = lookup(array, 3)
            found = point >= 0
            stacked = np.column_stack([nodes, np.where(found, point, 0)])
            finish('Triangles', array, found & (_tet_volume(coords, stacked, 0, 1, 2, 3) < 0.0))

        array = boundaries.get('Quadrilaterals')
        if array is not None and len(array):
            nodes, point = lookup(array, 4)
            found = point >= 0
            stacked = np.column_stack([nodes, np.where(found, point, 0)])
            failed = sum(
                (_tet_volume(coords, stacked, *tet) < 0.0).astype(int)
                for tet in ((0, 1, 2, 4), (1, 2, 3, 4), (2, 3, 0, 4), (3, 0, 1, 4))
            )
            finish('Quadrilaterals', array, found & (failed >= 3))

    return out


_SWAPS_BOUNDARY = {'Edges': [(0, 1)], 'Triangles': [(0, 2)], 'Quadrilaterals': [(1, 3)]}


def orient_mesh(mesh_data: dict, dims: tuple[int, ...] = (2, 3)) -> dict:
    r"""Orient interior and boundary elements the way SU2 expects them.

    Mirrors SU2's ``Check_IntElem_Orientation`` followed by ``Check_BoundElem_Orientation``, so SU2 finds
    nothing to re-orient on load.

    Args:
        mesh_data: Dictionary with keys coords, elements, boundaries.
        dims: Mesh dimensions to process. Others are returned unchanged.

    Returns:
        A shallow copy of ``mesh_data`` with new ``elements`` and ``boundaries`` entries. The input arrays are
        not modified.
    """
    coords = mesh_data['coords']
    if coords.shape[1] not in dims:
        return mesh_data
    elements = orient_elements(coords, mesh_data['elements'])
    boundaries = orient_boundaries(coords, elements, mesh_data['boundaries'])
    return {**mesh_data, 'elements': elements, 'boundaries': boundaries}
