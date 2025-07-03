from pathlib import Path

from numpy.typing import NDArray

from . import gmf, su2


def read_mesh(
    meshpath: str,
    markerpath: str | None = None,
    solpath: str | None = None,
    labelpath: str | None = None,
    read_sol: bool = False,
) -> tuple[NDArray, dict, dict] | tuple[NDArray, dict, dict, dict]:
    r"""Read a mesh file and return nodes and coordinates as numpy arrays.

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

        suffix = Path(meshpath).suffix
        if suffix in ['.mesh', '.meshb']:
            coords, elms, bnds = gmf.read_mesh(meshpath)
        elif suffix == '.su2':
            coords, elms, bnds = su2.read_mesh(meshpath, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        if read_sol and solpath is not None:
            labelpath = labelpath if labelpath is not None else ''
            num_point = coords.shape[0]
            dim = coords.shape[1]

            if suffix in ['.mesh', '.meshb']:
                sol = gmf.read_solution(solpath, num_point, dim, labelpath)
            elif suffix == '.su2':
                sol = su2.read_solution(solpath, num_point, dim, labelpath)

            return coords, elms, bnds, sol

        return coords, elms, bnds

    except Exception as e:
        print(f'Error reading mesh: {e}')
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
    write_sol: bool = False,
) -> bool:
    r"""Write nodes and coordinates to a meshb file.

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
        write_sol (bool, optional): Whether to write solution data. Defaults to False.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        markerpath = markerpath if markerpath is not None else ''

        suffix = Path(meshpath).suffix
        if suffix in ['.mesh', '.meshb']:
            success = gmf.write_mesh(meshpath, coords, elements, boundaries)
        elif suffix == '.su2':
            success = su2.write_mesh(meshpath, coords, elements, boundaries, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        if success and write_sol and solpath is not None and solution is not None:
            labelpath = labelpath if labelpath is not None else ''
            num_point = coords.shape[0]
            dim = coords.shape[1]

            if suffix in ['.mesh', '.meshb']:
                success = gmf.write_solution(solpath, solution, num_point, dim, labelpath)
            elif suffix == '.su2':
                success = su2.write_solution(solpath, solution, num_point, dim, labelpath)

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
    r"""Read solution data from a solution file.

    Args:
        solpath (str): Path to the solution file.
        num_point (int): Number of points/vertices.
        dim (int): Mesh dimension.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            return gmf.read_solution(solpath, num_point, dim, labelpath)
        elif suffix in ['.dat', '.csv']:
            return su2.read_solution(solpath, num_point, dim, labelpath)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .dat, and .csv'
            )
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
    """Write solution data to a solution file.

    Args:
        solpath (str): Path to the solution file.
        solution (dict[str, NDArray]): Dictionary of solution fields.
        num_point (int): Number of points/vertices.
        dim (int): Mesh dimension.
        labelpath (str, optional): Path to the map between solution strings and
                                   ref IDs. Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            return gmf.write_solution(solpath, solution, num_point, dim, labelpath)
        elif suffix in ['.dat', '.csv']:
            return su2.write_solution(solpath, solution, num_point, dim, labelpath)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .dat, and .csv'
            )
    except Exception as e:
        print(f'Error writing solution: {e}')
        return False
