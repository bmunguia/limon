from pathlib import Path

from numpy.typing import NDArray

from ..orientation import orient_mesh


def _get_libgmf():
    from . import libgmf
    return libgmf


def load_mesh(meshpath: Path | str) -> dict:
    r"""Read mesh data from a GMF mesh (.meshb) file.

    GMF files hold integer boundary references only, so no marker names are returned.

    Args:
        meshpath: Path to the mesh file.

    Returns:
        Dictionary with keys: coords, elements, boundaries, dim, num_point.
    """
    return _get_libgmf().load_mesh(str(meshpath))


def write_mesh(meshpath: Path | str, mesh_data: dict) -> bool:
    r"""Write mesh data to a GMF mesh (.meshb) file.

    2-D meshes are written with the interior and boundary orientation SU2 checks for. 3-D meshes are written as
    given, because GMF's 3-D element conventions are not verified against SU2's.

    Args:
        meshpath: Path to the mesh file.
        mesh_data: Dictionary containing mesh data with keys: coords, elements, boundaries.

    Returns:
        True if successful.
    """
    return _get_libgmf().write_mesh(str(meshpath), orient_mesh(mesh_data, dims=(2,)))


def load_solution(
    solpath: Path | str,
    num_ver: int,
    dim: int,
    names: list[str] | None = None,
) -> tuple[dict[str, NDArray], dict[int, str]]:
    r"""Read solution data from a GMF solution (.solb) file.

    Args:
        solpath: Path to the solution file.
        num_ver: Number of vertices.
        dim: Mesh dimension.
        names: Field names in file order; fields without a name are called REF_<n>.

    Returns:
        Tuple of (solution, label_map) where label_map maps the 1-based field index to its name.
    """
    return _get_libgmf().load_solution(str(solpath), num_ver, dim, list(names or []))


def write_solution(solpath: Path | str, solution: dict[str, NDArray], num_ver: int, dim: int) -> bool:
    r"""Write solution data to a GMF solution (.solb) file.

    Args:
        solpath: Path to the solution file.
        solution: Dictionary of solution fields, written in dict order.
        num_ver: Number of vertices.
        dim: Mesh dimension.

    Returns:
        True if successful.
    """
    return _get_libgmf().write_solution(str(solpath), solution, num_ver, dim)
