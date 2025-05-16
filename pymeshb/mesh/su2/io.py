from numpy.typing import NDArray

import pymeshb.mesh.su2.libsu2 as libsu2

def read_mesh(
    meshpath: str,
    solpath: str | None = None,
    read_sol: bool = False
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Read a SU2 mesh file and return nodes and element as numpy arrays.

    Args:
        meshpath (str): Path to the mesh file.
        solpath (str, optional): Path to the solution file. Defaults to None.
        read_sol (bool, optional): Whether to read solution data. Defaults to False.

    Returns:
        tuple: If read_sol=False, returns (coords, elements, boundaries)
               If read_sol=True, returns (coords, elements, boundaries, sol)
    """
    pass
    # try:
    #     solpath = solpath if solpath is not None else ''
    #     msh = libsu2.read_mesh(meshpath, solpath, read_sol)
    #
    #     if len(msh) == 4:
    #         coords, elms, bnds, sol = msh
    #         return coords, elms, bnds, sol
    #     else:
    #         return msh[0], msh[1], msh[2], {}
    #
    # except Exception as e:
    #     print(f"Error reading mesh: {e}")
    #     return None, None

def write_mesh(
    meshpath: str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    solpath: str | None = None,
    solution: dict[str, NDArray] | None= None,
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
        solpath (str, optional): Path to the solution file. Defaults to None.
        solution (dict, optional): Dictionary of solution data. Keys are
                                   field names, values are numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    pass
    # try:
    #     sol = solution if solution is not None else {}
    #     solpath = solpath if solpath is not None else ''
    #     success = libsu2.write_mesh(meshpath, coords, elements, boundaries, solpath, sol)
    #     return success

    # except Exception as e:
    #     print(f"Error writing mesh: {e}")
    #     return False
