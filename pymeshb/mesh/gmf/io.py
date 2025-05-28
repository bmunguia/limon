from numpy.typing import NDArray

from . import libgmf


def read_mesh(
    meshpath: str,
    solpath: str | None = None,
    labelpath: str | None = None,
    read_sol: bool = False
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read a meshb file and return nodes and element as numpy arrays.

    Args:
        meshpath (str): Path to the mesh file.
        solpath (str, optional): Path to the solution file. Defaults to None.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.
        read_sol (bool, optional): Whether to read solution data. Defaults to False.

    Returns:
        A tuple containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
        - solution: Dictionary of solution data (if read_sol is True, otherwise empty)
    """
    try:
        solpath = solpath if solpath is not None else ''
        coords, elms, bnds, sol = libgmf.read_mesh(meshpath, solpath, labelpath,
                                                   read_sol)

        return coords, elms, bnds, sol

    except Exception as e:
        print(f"Error reading mesh: {e}")
        return None, None, None, None


def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    solpath: str | None = None,
    labelpath: str | None = None,
    solution: dict[str, NDArray] | None = None,
) -> bool:
    r"""Write nodes and elements to a meshb file.

    Args:
        meshpath (str): Path to the mesh file.
        coords (numpy.ndarray): Coordinates of each node.
        elements (dict[str, NDArray]): Dictionary of mesh elements. Keys are
                                       element types, values are numpy arrays.
        boundaries (dict[str, NDArray]): Dictionary of mesh boundary elements.
                                         Keys are element types, values are
                                         numpy arrays.
        solpath (str, optional): Path to the solution file. Defaults to None.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.
        solution (dict, optional): Dictionary of solution data. Keys are
                                   field names, values are numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sol = solution if solution is not None else {}
        solpath = solpath if solpath is not None else ''
        success = libgmf.write_mesh(meshpath, coords, elements, boundaries,
                                    solpath, labelpath, sol)
        return success

    except Exception as e:
        print(f"Error writing mesh: {e}")
        return False
