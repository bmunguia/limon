from numpy.typing import NDArray

from . import libsu2


def read_mesh(
    meshpath: str,
    markerpath: str | None = None,
    solpath: str | None = None,
    labelpath: str | None = None,
    read_sol: bool = False
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read a SU2 mesh file and return nodes and element as numpy arrays.

    Args:
        meshpath (str): Path to the mesh file.
        markerpath (str, optional): Path to the map between marker strings and
                                    ref IDs. Defaults to None.
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
        markerpath = markerpath if markerpath is not None else ''
        solpath = solpath if solpath is not None else ''
        labelpath = labelpath if labelpath is not None else ''
        coords, elms, bnds, sol = libsu2.read_mesh(meshpath, markerpath, solpath,
                                                   labelpath, read_sol)

        return coords, elms, bnds, sol

    except Exception as e:
        print(f"Error reading mesh: {e}")
        return None, None, None, None


def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    markerpath: str | None = None,
    solpath: str | None = None,
    labelpath: str | None = None,
    solution: dict[str, NDArray] | None = None,
) -> bool:
    r"""Write nodes and elements to a SU2 mesh file.

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
        solpath (str, optional): Path to the solution file. Defaults to None.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.
        solution (dict, optional): Dictionary of solution data. Keys are
                                   field names, values are numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        markerpath = markerpath if markerpath is not None else ''
        solpath = solpath if solpath is not None else ''
        labelpath = labelpath if labelpath is not None else ''
        sol = solution if solution is not None else {}
        success = libsu2.write_mesh(meshpath, coords, elements, boundaries,
                                    markerpath, solpath, labelpath, sol)
        return success

    except Exception as e:
        print(f"Error writing mesh: {e}")
        return False
