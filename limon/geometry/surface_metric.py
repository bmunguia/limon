from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

_MODES = ('angle', 'hausdorff')


def surface_metric(
    mesh_data: dict,
    geodev: Mapping[str, float] | Iterable[tuple[str, float]],
    mode: str = 'angle',
    hmin: float = 1e-8,
    hmax: float = 1e8,
) -> NDArray:
    r"""Surface-geometry metric used as the background metric of a mesh adapter.

    Boundary nodes of the listed markers get a tangent edge length set by the local curvature
    and the normal size ``hmax``; all other nodes (interior, corners, unlisted markers) get the
    isotropic size ``hmax``. A node on several listed markers takes the first in ``geodev``.

    Args:
        mesh_data: Mesh dictionary from :func:`limon.mesh.load_mesh` (coords, boundaries, markers, dim).
        geodev: Ordered mapping (or pairs) of marker name to deviation. A deviation of 0 or less
                gives the size ``hmax``.
        mode: ``'angle'`` (deviation in degrees of boundary-normal turn per edge) or
              ``'hausdorff'`` (largest distance between the edge and the boundary).
        hmin: Smallest edge length.
        hmax: Largest edge length.

    Returns:
        C-contiguous float64 array of shape ``(num_point, d(d+1)/2)`` in limon's packed
        symmetric order (2-D: ``[xx, xy, yy]``), with every eigenvalue in ``[1/hmax^2, 1/hmin^2]``.

    Raises:
        NotImplementedError: For 3-D meshes.
        ValueError: For an unknown mode, invalid sizes, or a marker missing from ``mesh_data['markers']``.
    """
    mode = str(mode).lower()
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
    if mesh_data.get('dim') == 3:
        raise NotImplementedError(
            '3-D surface metrics are not yet implemented; the algorithm is in SU2 tag phd-greenlight, '
            'SU2_CFD/include/metrics/computeMetrics.hpp (geometricSurfaceMetrics).'
        )
    from . import _geometry

    pairs = list(geodev.items()) if isinstance(geodev, Mapping) else list(geodev)
    pairs = [(str(name), float(value)) for name, value in pairs]
    return np.ascontiguousarray(_geometry.surface_metric(mesh_data, pairs, mode, float(hmin), float(hmax)))
