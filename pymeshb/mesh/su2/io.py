from os import PathLike

from numpy.typing import NDArray

from . import libsu2


def load_mesh(
    meshpath: PathLike | str,
    markerpath: PathLike | str | None = None,
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read a SU2 mesh file and return nodes and element as numpy arrays.

    Args:
        meshpath: Path to the mesh file.
        markerpath: Path to the map between marker strings and
                    ref IDs. Defaults to None.

    Returns:
        A tuple containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
    """
    try:
        meshpath = str(meshpath)
        markerpath = str(markerpath) if markerpath is not None else ''
        coords, elms, bnds = libsu2.load_mesh(meshpath, markerpath)

        return coords, elms, bnds

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return None, None, None


def write_mesh(
    meshpath: PathLike | str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    markerpath: PathLike | str | None = None,
) -> bool:
    r"""Write mesh data to a SU2 mesh (.su2) file.

    Args:
        meshpath: Path to the mesh file.
        coords: Coordinates of each node.
        elements: Dictionary of mesh elements. Keys are
                  element types, values are numpy arrays.
        boundaries: Dictionary of mesh boundary elements.
                    Keys are element types, values are
                    numpy arrays.
        markerpath: Path to the map between marker strings and
                    ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        meshpath = str(meshpath)
        markerpath = str(markerpath) if markerpath is not None else ''
        success = libsu2.write_mesh(meshpath, coords, elements, boundaries, markerpath)
        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def load_solution(
    solpath: PathLike | str,
    num_point: int,
    dim: int,
    labelpath: PathLike | str | None = None,
) -> dict[str, NDArray]:
    r"""Read solution data from a SU2 solution file.

    Args:
        solpath: Path to the solution file.
        num_point: Number of points.
        dim: Mesh dimension.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        solpath = str(solpath)
        labelpath = str(labelpath) if labelpath is not None else ''
        return libsu2.load_solution(solpath, num_point, dim, labelpath)
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}


def write_solution(
    solpath: PathLike | str,
    solution: dict[str, NDArray],
    num_point: int,
    dim: int,
    labelpath: PathLike | str | None = None,
) -> bool:
    r"""Write solution data to a SU2 solution (.dat) file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields.
        num_point: Number of points.
        dim: Mesh dimension.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        solpath = str(solpath)
        labelpath = str(labelpath) if labelpath is not None else ''
        return libsu2.write_solution(solpath, solution, num_point, dim, labelpath)
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
