import numpy as np
from numpy.typing import NDArray

import libmeshb

def read_mesh(
    filepath: str,
    read_sol: bool = False
) -> tuple[NDArray, NDArray]:
    r"""Read a mesh file and return nodes and coordinates as numpy arrays.

    This function uses the C++ binding read_mesh from libmeshb.

    Args:
        filepath (str): Path to the mesh file
        read_sol (bool, optional): Whether to read solution fields. Defaults to False.

    Returns:
        tuple: If read_sol=False, returns (coords,)
               If read_sol=True, returns (coords, sol)
    """
    try:
        msh = libmeshb.read_mesh(filepath, read_sol)
        if len(msh) == 2:
            coords, sol = msh
            return coords, sol
        else:
            return msh[0], {}
    except Exception as e:
        print(f"Error reading mesh: {e}")
        return None, None

def write_mesh(
    filepath: str,
    coords: NDArray,
    solution: dict[str, NDArray] | None= None,
) -> bool:
    r"""Write nodes and coordinates to a mesh file.

    This function uses the C++ binding write_mesh from libmeshb.

    Args:
        filepath (str): Path to the mesh file
        coords (numpy.ndarray): Coordinates of each node
        solution (dict, optional): Dictionary of solution fields. Keys are
                                   field names, values are numpy arrays.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sol = solution if solution is not None else {}
        out = libmeshb.write_mesh(filepath, coords, sol)
        return out

    except Exception as e:
        print(f"Error writing mesh: {e}")
        return False
