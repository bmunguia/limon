from pathlib import Path
from os import PathLike

from numpy.typing import NDArray

from . import _ref_map, gmf, su2

RefMapKind = _ref_map.RefMapKind


def load_mesh(
    meshpath: PathLike | str,
    markerpath: PathLike | str | None = None,
    solpath: PathLike | str | None = None,
    labelpath: PathLike | str | None = None,
    read_sol: bool = False,
) -> tuple[NDArray, dict, dict] | tuple[NDArray, dict, dict, dict]:
    r"""Read a mesh file and return nodes and coordinates as numpy arrays.

    Args:
        meshpath: Path to the mesh file.
        markerpath: Path to the map between marker strings and
                    ref IDs. Defaults to None.
        solpath: Path to the solution file. Defaults to None.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.
        read_sol: Whether to read solution data. Defaults to False.

    Returns:
        A tuple containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
        - solution: Dictionary of solution data (if read_sol is True, otherwise empty)
    """
    try:
        suffix = Path(meshpath).suffix
        if suffix in ['.mesh', '.meshb']:
            coords, elms, bnds = gmf.load_mesh(meshpath)
        elif suffix == '.su2':
            coords, elms, bnds = su2.load_mesh(meshpath, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        if read_sol and solpath is not None:
            num_point = coords.shape[0]
            dim = coords.shape[1]

            if suffix in ['.mesh', '.meshb']:
                sol = gmf.load_solution(solpath, num_point, dim, labelpath)
            elif suffix == '.su2':
                sol = su2.load_solution(solpath, num_point, dim, labelpath)

            return coords, elms, bnds, sol

        return coords, elms, bnds

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return None, None, None, None


def write_mesh(
    meshpath: PathLike | str,
    coords: NDArray,
    elements: dict[str, NDArray],
    boundaries: dict[str, NDArray],
    markerpath: PathLike | str | None = None,
    solpath: PathLike | str | None = None,
    labelpath: PathLike | str | None = None,
    solution: dict[str, NDArray] | None = None,
    write_sol: bool = False,
) -> bool:
    r"""Write nodes and coordinates to a meshb file.

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
        solpath: Path to the solution file. Defaults to None.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.
        solution: Dictionary of solution data. Keys are
                  field names, values are numpy arrays.
        write_sol: Whether to write solution data. Defaults to False.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        suffix = Path(meshpath).suffix
        if suffix in ['.mesh', '.meshb']:
            success = gmf.write_mesh(meshpath, coords, elements, boundaries)
        elif suffix == '.su2':
            success = su2.write_mesh(meshpath, coords, elements, boundaries, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        if success and write_sol and solpath is not None and solution is not None:
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


def load_solution(
    solpath: PathLike | str,
    num_point: int,
    dim: int,
    labelpath: PathLike | str | None = None,
) -> dict[str, NDArray]:
    r"""Read solution data from a solution file.

    Args:
        solpath: Path to the solution file.
        num_point: Number of points/vertices.
        dim: Mesh dimension.
        labelpath: Path to the map between solution strings and
                   ref IDs. Defaults to None.

    Returns:
        dict[str, NDArray]: Dictionary of solution fields.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            return gmf.load_solution(solpath, num_point, dim, labelpath)
        elif suffix in ['.dat', '.csv']:
            return su2.load_solution(solpath, num_point, dim, labelpath)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .dat, and .csv'
            )
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
    """Write solution data to a solution file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields.
        num_point: Number of points/vertices.
        dim: Mesh dimension.
        labelpath: Path to the map between solution strings and
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


def load_ref_map(filename: PathLike | str) -> dict[int, str]:
    """Load reference map from file."""
    return _ref_map.load_ref_map(str(filename))


def write_ref_map(ref_map: dict[int, str], filename: PathLike | str, kind: _ref_map.RefMapKind) -> None:
    """Write reference map to file."""
    _ref_map.write_ref_map(ref_map, str(filename), kind)


def get_ref_name(ref_map: dict[int, str], ref_id: int) -> str:
    """Get reference name from map."""
    return _ref_map.get_ref_name(ref_map, ref_id)
