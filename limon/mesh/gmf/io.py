from os import PathLike

from numpy.typing import NDArray

from . import libgmf


def load_mesh(
    meshpath: PathLike | str,
) -> dict:
    r"""Read mesh data from a GMF mesh (.meshb) file.

    Args:
        meshpath: Path to the mesh file.

    Returns:
        Dictionary containing mesh data with keys:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
        - dim: Mesh dimension
        - num_point: Number of points
    """
    try:
        meshpath = str(meshpath)
        mesh_data = libgmf.load_mesh(meshpath)
        return mesh_data

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return {}


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
    labelpath: PathLike | str | None = None,
) -> dict[str, NDArray]:
    r"""Read solution data from a GMF solution (.solb) file.

    Args:
        solpath: Path to the solution file.
        num_ver: Number of vertices.
        dim: Mesh dimension.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        solpath = str(solpath)
        labelpath = str(labelpath) if labelpath is not None else ''
        return libgmf.load_solution(solpath, num_ver, dim, labelpath)
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}


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
