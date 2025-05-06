import numpy as np
from numpy.typing import NDArray

import libmeshb

def read_mesh(filepath: str) -> tuple[NDArray, NDArray]:
    r"""Read a mesh file and return nodes and coordinates as numpy arrays.

    This function uses the C++ binding read_mesh from libmeshb.

    Args:
        filepath (str): Path to the mesh file

    Returns:
        tuple: (nodes, coords) as numpy arrays
    """
    try:
        nodes, coords = libmeshb.read_mesh(filepath)
        return nodes, coords
    except Exception as e:
        print(f"Error reading mesh: {e}")
        return None, None

def write_mesh(nodes: NDArray, coords: NDArray, filepath: str) -> bool:
    r"""Write nodes and coordinates to a mesh file.

    This function uses the C++ binding write_mesh from libmeshb.

    Args:
        nodes (numpy.ndarray): Node IDs
        coords (numpy.ndarray): Coordinates of each node
        filepath (str): Path where to save the mesh

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        nodes_array = np.asarray(nodes, dtype=np.int32)
        coords_array = np.asarray(coords, dtype=np.float64)
        result = libmeshb.write_mesh(nodes_array, coords_array, filepath)

        if result:
            return True
        else:
            print(f"Failed to write mesh to {filepath}")
            return False
    except Exception as e:
        print(f"Error writing mesh: {e}")
        return False
