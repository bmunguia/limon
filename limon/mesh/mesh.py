from pathlib import Path
from os import PathLike

from numpy.typing import NDArray

from . import _ref_map, gmf, su2

RefMapKind = _ref_map.RefMapKind


def load_mesh_and_solution(
    meshpath: PathLike | str,
    solpath: PathLike | str,
    marker_map: dict[int, str] | None = None,
    label_map: dict[int, str] | None = None,
    read_markers: bool = False,
    read_labels: bool = False,
    markerpath: PathLike | str | None = None,
    labelpath: PathLike | str | None = None,
    write_markers: bool = False,
    write_labels: bool = False,
) -> dict:
    r"""Read a mesh file and solution file and return data as a dictionary.

    Args:
        meshpath: Path to the mesh file.
        solpath: Path to the solution file.
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.
        label_map: Dictionary mapping label IDs to names. Defaults to None.
        read_markers: Whether to read markers from markerpath file (GMF only). Defaults to False.
        read_labels: Whether to read labels from labelpath file (GMF only). Defaults to False.
        markerpath: Path to the marker reference map file. Defaults to None.
        labelpath: Path to the label reference map file. Defaults to None.
        write_markers: Whether to write the marker reference map (SU2 only). Defaults to False.
        write_labels: Whether to write the solution label reference map (SU2 only). Defaults to False.

    Returns:
        Dictionary containing:
        - coords: NDArray of node coordinates.
        - elements: Dictionary mapping element types to arrays of elements.
        - boundaries: Dictionary mapping boundary element types to arrays.
        - solution: Dictionary fieldsdata, 1 NDArray per scalar/vector/tensor.
        - dim: Mesh dimension.
        - num_point: Number of points.
        - markers: Dictionary mapping marker IDs to names.
        - labels: Dictionary mapping label IDs to names.
    """
    try:
        suffix = Path(meshpath).suffix.lower()
        if suffix in ['.mesh', '.meshb']:
            mesh_data, marker_map_out = gmf.load_mesh(meshpath, marker_map, read_markers, markerpath)
        elif suffix == '.su2':
            mesh_data, marker_map_out = su2.load_mesh(meshpath, write_markers, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        mesh_data['markers'] = marker_map_out

        num_point = mesh_data['num_point']
        dim = mesh_data['dim']

        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            sol, label_map_out = gmf.load_solution(solpath, num_point, dim, label_map, read_labels, labelpath)
        elif suffix in ['.csv', '.dat']:
            sol, label_map_out = su2.load_solution(solpath, num_point, dim, write_labels, labelpath)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
            )

        mesh_data['solution'] = sol
        mesh_data['labels'] = label_map_out
        return mesh_data

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return {}


def write_mesh_and_solution(
    meshpath: PathLike | str,
    solpath: PathLike | str,
    mesh_data: dict,
    marker_map: dict[int, str] | None = None,
) -> bool:
    r"""Write mesh and solution data to files.

    Args:
        meshpath: Path to the mesh file.
        solpath: Path to the solution file.
        mesh_data: Dictionary containing mesh and solution data with keys:
                   coords, elements, boundaries, solution, dim, num_points
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate required keys for mesh
        if 'coords' not in mesh_data:
            raise ValueError("mesh_data must contain 'coords' key")
        if 'elements' not in mesh_data:
            raise ValueError("mesh_data must contain 'elements' key")
        if 'boundaries' not in mesh_data:
            raise ValueError("mesh_data must contain 'boundaries' key")
        if 'solution' not in mesh_data:
            raise ValueError("mesh_data must contain 'solution' key")
        if 'dim' not in mesh_data:
            raise ValueError("mesh_data must contain 'dim' key")
        if 'num_points' not in mesh_data:
            raise ValueError("mesh_data must contain 'num_points' key")

        if marker_map is None:
            marker_map = {}

        # Write mesh
        suffix = Path(meshpath).suffix.lower()
        if suffix in ['.mesh', '.meshb']:
            success = gmf.write_mesh(meshpath, mesh_data)
        elif suffix == '.su2':
            success = su2.write_mesh(meshpath, mesh_data, marker_map)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        # Write solution
        num_point = mesh_data['num_points']
        dim = mesh_data['dim']
        solution = mesh_data['solution']

        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            success = gmf.write_solution(solpath, solution, num_point, dim)
        elif suffix in ['.csv', '.dat']:
            success = su2.write_solution(solpath, solution, num_point, dim)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
            )

        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def load_mesh(
    meshpath: PathLike | str,
    marker_map: dict[int, str] | None = None,
    read_markers: bool = False,
    markerpath: PathLike | str | None = None,
    write_markers: bool = False,
) -> dict:
    r"""Read a mesh file and return data as a dictionary.

    Args:
        meshpath: Path to the mesh file.
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.
        read_markers: Whether to read markers from markerpath file (GMF only). Defaults to False.
        markerpath: Path to the marker reference map file. Defaults to None.
        write_markers: Whether to write the marker reference map (SU2 only). Defaults to False.

    Returns:
        Dictionary containing:
        - coords: NDArray of node coordinates
        - elements: Dictionary mapping element types to arrays of elements
        - boundaries: Dictionary mapping boundary element types to arrays
        - dim: Mesh dimension
        - num_point: Number of points
        - markers: Dictionary mapping marker IDs to names
    """
    try:
        suffix = Path(meshpath).suffix.lower()
        if suffix in ['.mesh', '.meshb']:
            mesh_data, marker_map_out = gmf.load_mesh(meshpath, marker_map, read_markers, markerpath)
        elif suffix == '.su2':
            mesh_data, marker_map_out = su2.load_mesh(meshpath, write_markers, markerpath)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        mesh_data['markers'] = marker_map_out
        return mesh_data

    except Exception as e:
        print(f'Error reading mesh: {e}')
        return {}


def write_mesh(
    meshpath: PathLike | str,
    mesh_data: dict,
    marker_map: dict[int, str] | None = None,
) -> bool:
    r"""Write mesh data to a file.

    Args:
        meshpath: Path to the mesh file.
        mesh_data: Dictionary containing mesh data with keys:
                   coords, elements, boundaries
        marker_map: Dictionary mapping marker IDs to names. Defaults to None.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if marker_map is None:
            marker_map = {}

        suffix = Path(meshpath).suffix.lower()
        if suffix in ['.mesh', '.meshb']:
            success = gmf.write_mesh(meshpath, mesh_data)
        elif suffix == '.su2':
            success = su2.write_mesh(meshpath, mesh_data, marker_map)
        else:
            raise ValueError(f'Unsupported mesh file format: {suffix}. Supported formats are .mesh, .meshb, and .su2')

        return success

    except Exception as e:
        print(f'Error writing mesh: {e}')
        return False


def load_solution(
    solpath: PathLike | str,
    num_point: int,
    dim: int,
    label_map: dict[int, str] | None = None,
    read_labels: bool = False,
    labelpath: PathLike | str | None = None,
    write_labels: bool = False,
) -> dict:
    r"""Read solution data from a solution file.

    Args:
        solpath: Path to the solution file.
        num_point: Number of points/vertices.
        dim: Mesh dimension.
        label_map: Dictionary mapping label IDs to names. Defaults to None.
        read_labels: Whether to read labels from labelpath file (GMF only). Defaults to False.
        labelpath: Path to the label reference map file. Defaults to None.
        write_labels: Whether to write the solution label reference map (SU2 only). Defaults to False.

    Returns:
        Dictionary containing:
        - solution: Dictionary of solution fields, 1 NDArray per scalar/vector/tensor.
        - labels: Dictionary mapping label IDs to names.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            solution, label_map_out = gmf.load_solution(solpath, num_point, dim, label_map, read_labels, labelpath)
            return {'solution': solution, 'labels': label_map_out}
        elif suffix in ['.csv', '.dat']:
            solution, label_map_out = su2.load_solution(solpath, num_point, dim, write_labels, labelpath)
            return {'solution': solution, 'labels': label_map_out}
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
            )
    except Exception as e:
        print(f'Error reading solution: {e}')
        return {}


def write_solution(
    solpath: PathLike | str,
    solution: dict[str, NDArray],
    num_point: int,
    dim: int,
) -> bool:
    """Write solution data to a solution file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields,1 NDArray per scalar/vector/tensor.
        num_point: Number of points/vertices.
        dim: Mesh dimension.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        suffix = Path(solpath).suffix.lower()
        if suffix in ['.sol', '.solb']:
            return gmf.write_solution(solpath, solution, num_point, dim)
        elif suffix in ['.csv', '.dat']:
            return su2.write_solution(solpath, solution, num_point, dim)
        else:
            raise ValueError(
                f'Unsupported solution file format: {suffix}. Supported formats are .sol, .solb, .csv, and .dat'
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
