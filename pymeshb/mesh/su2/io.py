from numpy.typing import NDArray

from . import libsu2


def read_mesh(
    meshpath: str,
    markerpath: str | None = None,
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read a SU2 mesh file and return nodes and element as numpy arrays.

    Args:
        meshpath (str): Path to the mesh file.
        markerpath (str, optional): Path to the map between marker strings and
                                    ref IDs. Defaults to None.

    Returns:
        A tuple containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
    """
    try:
        markerpath = markerpath if markerpath is not None else ''
        coords, elms, bnds = libsu2.read_mesh(meshpath, markerpath)

        return coords, elms, bnds

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return None, None, None


def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    markerpath: str | None = None,
) -> bool:
    r"""Read mesh data from a SU2 mesh (.su2) file.

    Args:
        meshpath (str): Path to the mesh file.
        coords (numpy.ndarray): Coordinates of each node.
        elements (dict[str, NDArray]): Dictionary of mesh elements. Keys are
                                       element types, values are numpy arrays.
        boundaries (dict[str, NDArray]): Dictionary of mesh boundary elements.
                                         Keys are element types, values are
                                         numpy arrays.
        markerpath (str, optional): Path to the map between marker strings and
                                    ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        markerpath = markerpath if markerpath is not None else ''
        success = libsu2.write_mesh(meshpath, coords, elements, boundaries, markerpath)
        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def read_solution(
    solpath: str,
    num_point: int,
    dim: int,
    labelpath: str | None = None,
) -> dict[str, NDArray]:
    r"""Write mesh data to a SU2 mesh (.su2) file.

    Args:
        solpath (str): Path to the solution file.
        num_point (int): Number of points.
        dim (int): Mesh dimension.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        labelpath = labelpath if labelpath is not None else ''
        return libsu2.read_solution(solpath, num_point, dim, labelpath)
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}


def write_solution(
    solpath: str,
    solution: dict[str, NDArray],
    num_point: int,
    dim: int,
    labelpath: str | None = None,
) -> bool:
    r"""Write solution data to a SU2 solution (.dat) file.

    Args:
        solpath (str): Path to the solution file.
        solution (dict[str, NDArray]): Dictionary of solution fields.
        num_point (int): Number of points.
        dim (int): Mesh dimension.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        labelpath = labelpath if labelpath is not None else ''
        return libsu2.write_solution(solpath, solution, num_point, dim, labelpath)
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
