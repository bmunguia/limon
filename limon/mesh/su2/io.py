from pathlib import Path

from numpy.typing import NDArray

from ..orientation import orient_mesh


def _get_libsu2():
    from . import libsu2
    return libsu2


def load_mesh(meshpath: Path | str) -> tuple[dict, dict[int, str]]:
    r"""Read a SU2 mesh file.

    Args:
        meshpath: Path to the mesh file.

    Returns:
        Tuple of (mesh_data, marker_map) where:
        - mesh_data: Dictionary with keys: coords, elements, boundaries, dim, num_point
        - marker_map: Dictionary mapping marker IDs to names
    """
    return _get_libsu2().load_mesh(str(meshpath))


def write_mesh(meshpath: Path | str, mesh_data: dict, marker_map: dict[int, str] | None = None) -> bool:
    r"""Write mesh data to a SU2 mesh (.su2) file.

    Interior and boundary elements are written with the orientation SU2 checks for on load, so it finds nothing
    to re-orient.

    Args:
        meshpath: Path to the mesh file.
        mesh_data: Dictionary containing mesh data with keys: coords, elements, boundaries.
        marker_map: Dictionary mapping marker IDs to names. Defaults to mesh_data['markers'].

    Returns:
        True if successful.
    """
    if marker_map is None:
        marker_map = mesh_data.get('markers', {})
    mesh_data = orient_mesh(mesh_data)
    return _get_libsu2().write_mesh(str(meshpath), mesh_data, dict(marker_map))


def load_solution(solpath: Path | str, num_point: int, dim: int) -> tuple[dict[str, NDArray], dict[int, str]]:
    r"""Read solution data from a SU2 solution file.

    Args:
        solpath: Path to the solution file.
        num_point: Number of points.
        dim: Mesh dimension.

    Returns:
        Tuple of (solution, label_map) where label_map maps the 1-based field index to its name.
    """
    return _get_libsu2().load_solution(str(solpath), num_point, dim)


def write_solution(solpath: Path | str, solution: dict[str, NDArray], num_point: int, dim: int) -> bool:
    r"""Write solution data to a SU2 solution file (.dat is binary, anything else ASCII).

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields.
        num_point: Number of points.
        dim: Mesh dimension.

    Returns:
        True if successful.
    """
    return _get_libsu2().write_solution(str(solpath), solution, num_point, dim)
