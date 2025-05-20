from numpy.typing import NDArray

from pymeshb.mesh import gmf, su2


def read_mesh(
    meshpath: str,
    solpath: str | None = None,
    read_sol: bool = False
) -> tuple[NDArray, dict, dict, dict]:
    r"""Read a mesh file and return nodes and coordinates as numpy arrays.

    Args:
        meshpath (str): Path to the mesh file.
        solpath (str, optional): Path to the solution file. Defaults to None.
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
        if 'mesh' in meshpath or 'meshb' in meshpath:
            msh = gmf.read_mesh(meshpath, solpath, read_sol)
        elif 'su2' in meshpath:
            msh = su2.read_mesh(meshpath, solpath, read_sol)
        else:
            raise ValueError(
                f'Unsupported mesh file format: {meshpath}. '
                'Supported formats are .mesh, .meshb, and .su2'
            )

        return msh

    except Exception as e:
        print(f"Error reading mesh: {e}")
        return None, None, None, None


def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    solpath: str | None = None,
    solution: dict[str, NDArray] | None = None,
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
        solpath (str, optional): Path to the solution file. Defaults to None.
        solution (dict, optional): Dictionary of solution data. Keys are
                                   field names, values are numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sol = solution if solution is not None else {}
        solpath = solpath if solpath is not None else ''
        if 'mesh' in meshpath or 'meshb' in meshpath:
            success = gmf.write_mesh(meshpath, coords, elements, boundaries, solpath, sol)
        elif 'su2' in meshpath:
            success = su2.write_mesh(meshpath, coords, elements, boundaries, solpath, sol)
        else:
            raise ValueError(
                f'Unsupported mesh file format: {meshpath}. '
                'Supported formats are .mesh, .meshb, and .su2'
            )

        return success

    except Exception as e:
        print(f"Error writing mesh: {e}")
        return False
