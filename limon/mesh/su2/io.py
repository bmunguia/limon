from pathlib import Path

from numpy.typing import NDArray

from . import libsu2


def load_mesh(
    meshpath: Path | str,
    write_markers: bool = False,
    markerpath: Path | str | None = None,
) -> tuple[dict, dict[int, str]]:
    r"""Read a SU2 mesh file and return nodes and element as numpy arrays.

    Args:
        meshpath: Path to the mesh file.
        write_markers: Whether to write the marker reference map.
                       Defaults to False.
        markerpath: Path to write the marker reference map file.
                    Defaults to None.

    Returns:
        Tuple of (mesh_data, marker_map) where:
        - mesh_data: Dictionary with keys: coords, elements, boundaries, dim, num_point
        - marker_map: Dictionary mapping marker IDs to names
    """
    try:
        meshpath = str(meshpath)
        markerpath = str(markerpath) if markerpath is not None else ''
        mesh_data, marker_map = libsu2.load_mesh(meshpath, write_markers, markerpath)
        return mesh_data, marker_map

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return {}, {}


def write_mesh(
    meshpath: Path | str,
    mesh_data: dict,
    marker_map: dict[int, str] | None = None,
) -> bool:
    r"""Write mesh data to a SU2 mesh (.su2) file.

    Args:
        meshpath: Path to the mesh file.
        mesh_data: Dictionary containing mesh data with keys:
                   coords, elements, boundaries
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.
                    If None, will use mesh_data['markers'] if available.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        meshpath = str(meshpath)
        if marker_map is None:
            # Fallback to mesh_data['markers'] if available
            marker_map = mesh_data.get('markers', {})
        success = libsu2.write_mesh(meshpath, mesh_data, marker_map)
        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def load_solution(
    solpath: Path | str,
    num_point: int,
    dim: int,
    write_labels: bool = False,
    labelpath: Path | str | None = None,
) -> tuple[dict[str, NDArray], dict[int, str]]:
    r"""Read solution data from a SU2 solution file.

    Args:
        solpath: Path to the solution file.
        num_point: Number of points.
        dim: Mesh dimension.
        write_labels: Whether to write the solution label reference map.
                      Defaults to False.
        labelpath: Path to write the label reference map file.
                   Defaults to None.

    Returns:
        Tuple of (solution, label_map) where:
        - solution: Dictionary of solution fields
        - label_map: Dictionary mapping label IDs to names
    """
    try:
        solpath = str(solpath)
        labelpath = str(labelpath) if labelpath is not None else ''
        solution, label_map = libsu2.load_solution(solpath, num_point, dim, write_labels, labelpath)
        return solution, label_map
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}, {}


def write_solution(
    solpath: Path | str,
    solution: dict[str, NDArray],
    num_point: int,
    dim: int,
) -> bool:
    r"""Write solution data to a SU2 solution (.dat) file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields.
        num_point: Number of points.
        dim: Mesh dimension.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        solpath = str(solpath)
        return libsu2.write_solution(solpath, solution, num_point, dim)
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
