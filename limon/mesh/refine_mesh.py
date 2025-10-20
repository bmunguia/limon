from .refine import _refine


def refine_mesh(mesh_data: dict) -> dict:
    """Refine a mesh uniformly.

    Args:
        mesh_data: Dictionary containing mesh data with keys:
                   coords, elements, boundaries, dim, num_point

    Returns:
        Dictionary containing refined mesh data with the same keys.

    Raises:
        ValueError: If mesh dimension is not supported.
        RuntimeError: If refinement fails.
    """
    if 'dim' not in mesh_data:
        raise ValueError("mesh_data must contain 'dim' key")

    dim = mesh_data['dim']

    if dim == 2:
        return _refine.refine_2d(mesh_data)
    elif dim == 3:
        raise NotImplementedError('3D mesh refinement is not yet implemented')
    else:
        raise ValueError(f'Unsupported mesh dimension: {dim}. Must be 2 or 3.')
