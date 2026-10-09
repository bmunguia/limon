from pathlib import Path

from numpy.typing import NDArray

from . import gmf, su2
from .names import LimonIOError, read_names, update_names

_GMF_MESH = ('.mesh', '.meshb')
_SU2_MESH = ('.su2',)
_GMF_SOL = ('.sol', '.solb')
_SU2_SOL = ('.csv', '.dat')


def _fail(action: str, path: Path | str, error: Exception) -> LimonIOError:
    return LimonIOError(f'Failed to {action} {path}: {error}')


def _check_mesh_data(mesh_data: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in mesh_data:
            raise ValueError(f"mesh_data must contain '{key}' key")


def load_mesh(meshpath: Path | str, marker_map: dict[int, str] | None = None) -> dict:
    r"""Read a mesh file and return its data as a dictionary.

    Args:
        meshpath: Path to the mesh file (.su2, .mesh or .meshb).
        marker_map: Marker names for GMF meshes, which store integer references only.
                    Defaults to the ``<stem>.names.json`` sidecar, if present. SU2 meshes
                    carry their own names and ignore it. The dictionary is not modified.

    Returns:
        Dictionary containing:
        - coords: NDArray of node coordinates.
        - elements: Dictionary mapping element types to arrays of elements.
        - boundaries: Dictionary mapping boundary element types to arrays, plus ``Corners``
          (node indices) when the file has them.
        - dim: Mesh dimension.
        - num_point: Number of points.
        - markers: Dictionary mapping marker IDs to names.

    Raises:
        LimonIOError: If the file cannot be read.
    """
    try:
        suffix = Path(meshpath).suffix.lower()
        if suffix in _GMF_MESH:
            mesh_data = gmf.load_mesh(meshpath)
            markers = dict(marker_map) if marker_map else read_names(meshpath).get('markers', {})
        elif suffix in _SU2_MESH:
            mesh_data, markers = su2.load_mesh(meshpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')
    except Exception as e:
        raise _fail('read mesh', meshpath, e) from e

    mesh_data['markers'] = dict(markers)
    return mesh_data


def write_mesh(
    meshpath: Path | str,
    mesh_data: dict,
    marker_map: dict[int, str] | None = None,
    write_names: bool = True,
) -> bool:
    r"""Write mesh data to a file.

    Args:
        meshpath: Path to the mesh file (.su2, .mesh or .meshb).
        mesh_data: Dictionary containing mesh data with keys: coords, elements, boundaries.
        marker_map: Dictionary mapping marker IDs to names. Defaults to ``mesh_data['markers']``.
        write_names: Whether GMF output also writes the marker names to the ``<stem>.names.json``
                     sidecar. SU2 meshes always carry the names.

    Returns:
        True if successful.

    Raises:
        LimonIOError: If the file cannot be written.
    """
    try:
        _check_mesh_data(mesh_data, ('coords', 'elements', 'boundaries'))
        markers = marker_map if marker_map is not None else mesh_data.get('markers', {})
        suffix = Path(meshpath).suffix.lower()
        if suffix in _GMF_MESH:
            gmf.write_mesh(meshpath, mesh_data)
            if write_names and markers:
                update_names(meshpath, markers=markers)
        elif suffix in _SU2_MESH:
            su2.write_mesh(meshpath, mesh_data, markers)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')
        return True
    except Exception as e:
        raise _fail('write mesh', meshpath, e) from e


def load_solution(
    solpath: Path | str,
    num_point: int,
    dim: int,
    names: list[str] | None = None,
) -> dict:
    r"""Read solution data from a solution file.

    Args:
        solpath: Path to the solution file (.sol, .solb, .csv or .dat).
        num_point: Number of points/vertices.
        dim: Mesh dimension.
        names: Field names in file order for GMF solutions, which store no names. Defaults to the
               ``<stem>.names.json`` sidecar, then ``REF_<n>``. SU2 files name their own fields.

    Returns:
        Dictionary containing:
        - solution: Dictionary of solution fields, 1 NDArray per scalar/vector/tensor.
        - labels: Dictionary mapping the 1-based field index to the field name.

    Raises:
        LimonIOError: If the file cannot be read.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in _GMF_SOL:
            if names is None:
                names = read_names(solpath).get('labels')
            solution, labels = gmf.load_solution(solpath, num_point, dim, names)
        elif suffix in _SU2_SOL:
            solution, labels = su2.load_solution(solpath, num_point, dim)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
            )
        return {'solution': solution, 'labels': dict(labels)}
    except Exception as e:
        raise _fail('read solution', solpath, e) from e


def write_solution(
    solpath: Path | str,
    solution: dict[str, NDArray],
    num_point: int,
    dim: int,
    write_names: bool = True,
) -> bool:
    """Write solution data to a solution file.

    Args:
        solpath: Path to the solution file (.sol, .solb, .csv or .dat).
        solution: Dictionary of solution fields, 1 NDArray per scalar/vector/tensor, written in dict order.
        num_point: Number of points/vertices.
        dim: Mesh dimension.
        write_names: Whether GMF output also writes the field names to the ``<stem>.names.json`` sidecar.

    Returns:
        True if successful.

    Raises:
        LimonIOError: If the file cannot be written.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in _GMF_SOL:
            gmf.write_solution(solpath, solution, num_point, dim)
            if write_names and solution:
                update_names(solpath, labels=list(solution))
        elif suffix in _SU2_SOL:
            su2.write_solution(solpath, solution, num_point, dim)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
            )
        return True
    except Exception as e:
        raise _fail('write solution', solpath, e) from e


def load_mesh_and_solution(
    meshpath: Path | str,
    solpath: Path | str,
    marker_map: dict[int, str] | None = None,
    names: list[str] | None = None,
) -> dict:
    r"""Read a mesh file and a solution file and return their data as one dictionary.

    Args:
        meshpath: Path to the mesh file.
        solpath: Path to the solution file.
        marker_map: Marker names for GMF meshes (see :func:`load_mesh`).
        names: Field names in file order for GMF solutions (see :func:`load_solution`).

    Returns:
        The :func:`load_mesh` dictionary plus:
        - solution: Dictionary of fields, 1 NDArray per scalar/vector/tensor.
        - labels: Dictionary mapping the 1-based field index to the field name.

    Raises:
        LimonIOError: If either file cannot be read.
    """
    mesh_data = load_mesh(meshpath, marker_map)
    mesh_data.update(load_solution(solpath, mesh_data['num_point'], mesh_data['dim'], names))
    return mesh_data


def write_mesh_and_solution(
    meshpath: Path | str,
    solpath: Path | str,
    mesh_data: dict,
    marker_map: dict[int, str] | None = None,
    write_names: bool = True,
) -> bool:
    r"""Write mesh and solution data to files.

    Args:
        meshpath: Path to the mesh file.
        solpath: Path to the solution file.
        mesh_data: Dictionary containing mesh and solution data with keys:
                   coords, elements, boundaries, solution, dim, num_point
        marker_map: Dictionary mapping marker IDs to names. Defaults to ``mesh_data['markers']``.
        write_names: Whether GMF output writes the ``<stem>.names.json`` sidecar.

    Returns:
        True if successful.

    Raises:
        LimonIOError: If either file cannot be written.
    """
    try:
        _check_mesh_data(mesh_data, ('coords', 'elements', 'boundaries', 'solution', 'dim', 'num_point'))
    except ValueError as e:
        raise _fail('write mesh', meshpath, e) from e
    write_mesh(meshpath, mesh_data, marker_map, write_names)
    return write_solution(solpath, mesh_data['solution'], mesh_data['num_point'], mesh_data['dim'], write_names)
