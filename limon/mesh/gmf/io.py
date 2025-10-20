from os import PathLike

from numpy.typing import NDArray

from . import libgmf


def load_mesh(
    meshpath: PathLike | str,
    marker_map: dict[int, str] | None = None,
    read_markers: bool = False,
    markerpath: PathLike | str | None = None,
) -> tuple[dict, dict[int, str]]:
    r"""Read mesh data from a GMF mesh (.meshb) file.

    Args:
        meshpath: Path to the mesh file.
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.
        read_markers: Whether to read markers from markerpath file. Defaults to False.
        markerpath: Path to the marker reference map file. Defaults to None.

    Returns:
        Tuple of (mesh_data, marker_map) where:
        - mesh_data: Dictionary with keys: coords, elements, boundaries, dim, num_point
        - marker_map: Dictionary mapping marker IDs to names
    """
    try:
        meshpath = str(meshpath)
        markerpath = str(markerpath) if markerpath is not None else ''
        if marker_map is None:
            marker_map = {}
        mesh_data, marker_map = libgmf.load_mesh(meshpath, marker_map, read_markers, markerpath)
        return mesh_data, marker_map

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return {}, {}


def write_mesh(
    meshpath: PathLike | str,
    mesh_data: dict,
) -> bool:
    r"""Write mesh data to a GMF mesh (.meshb) file.

    Args:
        meshpath: Path to the mesh file.
        mesh_data: Dictionary containing mesh data with keys:
                   coords, elements, boundaries

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        meshpath = str(meshpath)
        success = libgmf.write_mesh(meshpath, mesh_data)
        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def load_solution(
    solpath: PathLike | str,
    num_ver: int,
    dim: int,
    label_map: dict[int, str] | None = None,
    read_labels: bool = False,
    labelpath: PathLike | str | None = None,
) -> tuple[dict[str, NDArray], dict[int, str]]:
    r"""Read solution data from a GMF solution (.solb) file.

    Args:
        solpath: Path to the solution file.
        num_ver: Number of vertices.
        dim: Mesh dimension.
        label_map: Dictionary mapping label IDs to names. Defaults to None.
        read_labels: Whether to read labels from labelpath file. Defaults to False.
        labelpath: Path to the label reference map file. Defaults to None.

    Returns:
        Tuple of (solution, label_map) where:
        - solution: Dictionary of solution fields
        - label_map: Dictionary mapping label IDs to names
    """
    try:
        solpath = str(solpath)
        labelpath = str(labelpath) if labelpath is not None else ''
        if label_map is None:
            label_map = {}
        solution, label_map = libgmf.load_solution(solpath, num_ver, dim, label_map, read_labels, labelpath)
        return solution, label_map
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}, {}


def write_solution(
    solpath: PathLike | str,
    solution: dict[str, NDArray],
    num_ver: int,
    dim: int,
) -> bool:
    r"""Write solution data to a GMF solution (.solb) file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields.
        num_ver: Number of vertices.
        dim: Mesh dimension.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        solpath = str(solpath)
        return libgmf.write_solution(solpath, solution, num_ver, dim)
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
