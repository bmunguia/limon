from numpy.typing import NDArray

from . import libgmf


def read_mesh(
    meshpath: str,
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read mesh data from a GMF mesh (.meshb) file.

    Args:
        meshpath (str): Path to the mesh file.

    Returns:
        A tuple containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
    """
    try:
        coords, elms, bnds = libgmf.read_mesh(meshpath)

        return coords, elms, bnds

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return None, None, None


def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
) -> bool:
    r"""Write mesh data to a GMF mesh (.meshb) file.

    Args:
        meshpath (str): Path to the mesh file.
        coords (numpy.ndarray): Coordinates of each node.
        elements (dict[str, NDArray]): Dictionary of mesh elements. Keys are
                                       element types, values are numpy arrays.
        boundaries (dict[str, NDArray]): Dictionary of mesh boundary elements.
                                         Keys are element types, values are
                                         numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        success = libgmf.write_mesh(meshpath, coords, elements, boundaries)
        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False
    
def read_solution(
    solpath: str,
    num_ver: int,
    dim: int,
    labelpath: str | None = None,
) -> dict[str, NDArray]:
    r"""Read solution data from a GMF solution (.solb) file.

    Args:
        solpath (str): Path to the solution file.
        labelpath (str, optional): Path to the map between solution strings and
                                ref IDs. Defaults to None.
        num_ver (int): Number of vertices.
        dim (int): Mesh dimension. Defaults to 3.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        labelpath = labelpath if labelpath is not None else ''
        return libgmf.read_solution(solpath, num_ver, dim, labelpath)
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}


def write_solution(
    solpath: str,
    solution: dict[str, NDArray],
    num_ver: int,
    dim: int,
    labelpath: str | None = None,
) -> bool:
    r"""Write solution data to a GMF solution (.solb) file.

    Args:
        solpath (str): Path to the solution file.
        solution (dict[str, NDArray]): Dictionary of solution fields.
        num_ver (int): Number of vertices.
        dim (int): Mesh dimension.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        labelpath = labelpath if labelpath is not None else ''
        return libgmf.write_solution(solpath, solution, num_ver, dim, labelpath)
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
