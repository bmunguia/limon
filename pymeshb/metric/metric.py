import numpy as np
from numpy.typing import NDArray

from . import _metric


def decompose(lower_tri: NDArray) -> tuple[NDArray, NDArray]:
    r"""Diagonalize a symmetric tensor represented by its lower triangular elements.

    This function computes the eigenvalues and eigenvectors of a symmetric tensor
    using the provided lower triangular elements.

    Args:
        lower_tri (numpy.ndarray): Lower triangular elements of the symmetric tensor.
            For 1D: [a[0][0]]
            For 2D: [a[0][0], a[1][0], a[1][1]]
            For 3D: [a[0][0], a[1][0], a[1][1], a[2][0], a[2][1], a[2][2]]

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - eigenvalues: Array of eigenvalues in descending order
            - eigenvectors: Array of eigenvectors as columns

    Raises:
        RuntimeError: If the input size is not compatible with 1D, 2D, or 3D tensor
        RuntimeError: If the symmetric tensor has negative eigenvalues (2D case)

    Examples:
        >>> # Diagonalize a 2D tensor
        >>> lower_tri = np.array([3.0, 1.0, 2.0])  # [a00, a10, a11]
        >>> eigenvalues, eigenvectors = decompose(lower_tri)
    """
    return _metric.decompose(np.asarray(lower_tri, dtype=np.float64))


def recompose(eigenvalues: NDArray, eigenvectors: NDArray) -> NDArray:
    r"""Recombine eigenvalues and eigenvectors to reconstruct the tensor.

    This function reconstructs the original symmetric tensor from its eigendecomposition
    using the formula M = V * D * V^T, where V contains eigenvectors and D is a diagonal
    matrix of eigenvalues.

    Args:
        eigenvalues (numpy.ndarray): Array of eigenvalues
        eigenvectors (numpy.ndarray): Array of eigenvectors as columns

    Returns:
        numpy.ndarray: Lower triangular elements of the reconstructed tensor.
            For 1D: [a[0][0]]
            For 2D: [a[0][0], a[1][0], a[1][1]]
            For 3D: [a[0][0], a[1][0], a[1][1], a[2][0], a[2][1], a[2][2]]

    Raises:
        RuntimeError: If dimensions of eigenvalues and eigenvectors don't match
        RuntimeError: If dimension is not 1, 2, or 3

    Examples:
        >>> # Reconstruct a tensor from its eigendecomposition
        >>> eigenvalues = np.array([4.0, 1.0])
        >>> eigenvectors = np.array([[0.8944, -0.4472], [0.4472, 0.8944]])
        >>> tensor = recompose(eigenvalues, eigenvectors)
    """
    return _metric.recompose(np.asarray(eigenvalues, dtype=np.float64), np.asarray(eigenvectors, dtype=np.float64))


def perturb(
    eigenvalues: NDArray, eigenvectors: NDArray, delta_eigenvals: NDArray, rotation_angles: NDArray
) -> tuple[NDArray, NDArray]:
    r"""Apply perturbations to eigenvalues and eigenvectors.

    This function applies perturbations to eigenvalues and eigenvectors and
    returns the perturbed eigenpairs. The perturbed eigenvectors are
    orthonormalized using the Gram-Schmidt process.

    Args:
        eigenvalues (numpy.ndarray): Original eigenvalues
        eigenvectors (numpy.ndarray): Original eigenvectors as columns
        delta_eigenvals (numpy.ndarray): Logarithmic perturbation values for eigenvalues
        rotation_angles (numpy.ndarray): Rotation angles in radians for eigenvector rotation.
                                        For 2D: Single angle for rotation in xy-plane.
                                        For 3D: Three angles for rotations in xy, yz, and xz planes.

    Raises:
        RuntimeError: If dimensions of inputs don't match
        RuntimeError: If dimension is not 1, 2, or 3
        RuntimeError: If rotation_angles doesn't have expected size (0 for 1D, 1 for 2D, 3 for 3D)

    Examples:
        >>> # Apply perturbations to eigenvalues and eigenvectors
        >>> eigenvalues = np.array([5.0, 3.0, 1.0])
        >>> eigenvectors = np.eye(3)  # Identity matrix as example
        >>> delta_vals = np.array([0.1, 0.0, -0.1])
        >>> rot_angles = np.array([0.05, 0.02, 0.03])  # xy, yz, xz rotations
        >>> pert_vals, pert_vecs = perturb(eigenvalues, eigenvectors, delta_vals, rot_angles)
    """
    return _metric.perturb(
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.float64),
        np.asarray(delta_eigenvals, dtype=np.float64),
        np.asarray(rotation_angles, dtype=np.float64),
    )


def perturb_metric_field(metrics: NDArray, delta_eigenvals: NDArray, rotation_angles: NDArray) -> NDArray:
    r"""Perturb a field of metric tensors.

    For each tensor in the input array, this function:
    1. Diagonalizes the tensor into eigenvalues and eigenvectors
    2. Applies specified perturbations to the eigenvalues and eigenvectors
    3. Recombines into a perturbed tensor

    Args:
        metrics (numpy.ndarray): Array of shape (num_point, num_met) where each row
                                contains the lower triangular elements of a tensor.
        delta_eigenvals (numpy.ndarray): Array of shape (num_point, dim) containing
                                        logarithmic eigenvalue perturbations for each point.
                                        dim is 1, 2, or 3 for 1D, 2D, or 3D tensors.
        rotation_angles (numpy.ndarray): Array of shape (num_point, num_angle) containing
                                        rotation angles in radians for each point.
                                        num_angle is 0 for 1D, 1 for 2D, or 3 for 3D.

    Returns:
        numpy.ndarray: Array of perturbed metric tensors in lower triangular form.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements
        RuntimeError: If rotation_angles shape is not (num_point, num_angle) where
                     num_angle is 0 for 1D, 1 for 2D, or 3 for 3D

    Examples:
        >>> # Process a field of tensors with perturbations
        >>> metrics = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # First 3D tensor in lower triangular form
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> val_pert = np.array([
        ...     [0.1, 0.0, -0.1],  # Perturbations for first tensor's eigenvalues
        ...     [0.2, -0.05, 0.0]  # Perturbations for second tensor's eigenvalues
        ... ])
        >>> rot_angles = np.array([
        ...     [0.05, 0.02, 0.01],  # Rotations for first tensor (xy, yz, xz)
        ...     [0.03, 0.04, 0.02]   # Rotations for second tensor
        ... ])
        >>> perturbed_metrics = perturb_metric_field(metrics, val_pert, rot_angles)
    """
    return _metric.perturb_metric_field(
        np.asarray(metrics, dtype=np.float64),
        np.asarray(delta_eigenvals, dtype=np.float64),
        np.asarray(rotation_angles, dtype=np.float64),
    )


def integrate_metric_field(
    metrics: NDArray,
    volumes: NDArray,
    norm: int = 2,
) -> float:
    r"""Integrate the determinant of metric tensors over volumes.

    For each tensor in the input array, this function:
    1. Reconstructs the full tensor from lower triangular elements
    2. Computes the determinant of the tensor
    3. Multiplies by the corresponding volume and adds to the integral

    Args:
        metrics (numpy.ndarray): Array of shape (num_point, num_met) where each row
                                contains the lower triangular elements of a tensor.
                                num_met is 1, 3, or 6 for 1D, 2D, or 3D tensors.
        volumes (numpy.ndarray): Array of shape (num_point,) containing volumes for
                                each point.
        norm (int): Choice of norm for Lp-norm normalization.

    Returns:
        float: Scalar value representing the integrated determinant ∑(det(M_i) * volume_i).

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements
        RuntimeError: If volumes shape is not (num_point,)

    Examples:
        >>> # Integrate determinants of a field of 2D tensors
        >>> metrics = np.array([
        ...     [2.0, 0.5, 1.0],  # First 2D tensor [a00, a10, a11]
        ...     [3.0, 0.0, 1.0],  # Second 2D tensor
        ...     [1.5, 0.2, 2.0],  # Third 2D tensor
        ... ])
        >>> volumes = np.array([0.1, 0.2, 0.15])  # Volume for each point
        >>> integral = integrate_metric_field(metrics, volumes, norm=2)

        >>> # Integrate determinants of a field of 3D tensors
        >>> metrics_3d = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # 3D tensor [a00, a10, a11, a20, a21, a22]
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> volumes_3d = np.array([0.05, 0.08])
        >>> integral_3d = integrate_metric_field(metrics_3d, volumes_3d, norm=2)
    """
    return _metric.integrate_metric_field(
        np.asarray(metrics, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        norm,
    )


def normalize_metric_field(
    metrics: NDArray,
    metric_integral: float,
    complexity: int,
    norm: float,
) -> NDArray:
    r"""Normalize a field of metric tensors.

    For each tensor in the input array, this function:
    1. Reconstructs the full tensor from lower triangular elements
    2. Computes the determinant of the tensor
    3. Scales the tensor by the normalization factor

    The normalization applies the scaling factor:
    (complexity / metric_integral)^(2/dim) * det(M)^(-1/(2*norm + dim))

    Args:
        metrics (numpy.ndarray): Array of shape (num_point, num_met) where each row
                                contains the lower triangular elements of a tensor.
                                num_met is 1, 3, or 6 for 1D, 2D, or 3D tensors.
        metric_integral (float): Integral of metric determinants over the domain.
        complexity (int): Target complexity for normalization.
        norm (int): Choice of norm for Lp-norm normalization.

    Returns:
        numpy.ndarray: Array of normalized metric tensors in lower triangular form.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements

    Examples:
        >>> # Normalize a field of 2D tensors
        >>> metrics = np.array([
        ...     [2.0, 0.5, 1.0],  # First 2D tensor [a00, a10, a11]
        ...     [3.0, 0.0, 1.0],  # Second 2D tensor
        ...     [1.5, 0.2, 2.0],  # Third 2D tensor
        ... ])
        >>> norm = 2
        >>> metric_integral = 15.5  # Previously computed integral
        >>> complexity = 1000.0    # Target complexity
        >>> normalized = normalize_metric_field(metrics, metric_integral, complexity, norm)

        >>> # Normalize a field of 3D tensors
        >>> metrics_3d = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # 3D tensor
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> normalized_3d = normalize_metric_field(metrics_3d, metric_integral=25.8,
                                                   complexity=5000, norm=2)
    """
    return _metric.normalize_metric_field(
        np.asarray(metrics, dtype=np.float64),
        metric_integral,
        complexity,
        norm,
    )


def decompose_metric_field(metrics: NDArray) -> tuple[NDArray, NDArray]:
    r"""Decompose a field of metric tensors.

    For each tensor in the input array, this function:
    1. Diagonalizes the tensor into eigenvalues and eigenvectors
    2. Returns the eigenvalues and eigenvectors for all points

    Args:
        metrics (numpy.ndarray): Array of shape (num_point, num_met) where each row
                                contains the lower triangular elements of a tensor.
                                num_met is 1, 3, or 6 for 1D, 2D, or 3D tensors.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - eigenvalues: Array of shape (num_point, dim) with eigenvalues for each point
            - eigenvectors: Array of shape (num_point, dim, dim) with eigenvectors for each point

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements

    Examples:
        >>> # Decompose a field of 2D tensors
        >>> metrics = np.array([
        ...     [2.0, 0.5, 1.0],  # First 2D tensor [a00, a10, a11]
        ...     [3.0, 0.0, 1.0],  # Second 2D tensor
        ...     [1.5, 0.2, 2.0],  # Third 2D tensor
        ... ])
        >>> eigenvalues, eigenvectors = decompose_metric_field(metrics)
        >>> print(eigenvalues.shape)  # (3, 2)
        >>> print(eigenvectors.shape)  # (3, 2, 2)

        >>> # Decompose a field of 3D tensors
        >>> metrics_3d = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # 3D tensor
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> eigenvals_3d, eigenvecs_3d = decompose_metric_field(metrics_3d)
        >>> print(eigenvals_3d.shape)  # (2, 3)
        >>> print(eigenvecs_3d.shape)  # (2, 3, 3)
    """
    return _metric.decompose_metric_field(np.asarray(metrics, dtype=np.float64))


def recompose_metric_field(eigenvalues: NDArray, eigenvectors: NDArray) -> NDArray:
    r"""Recompose a field of metric tensors from eigenvalues and eigenvectors.

    For each set of eigenvalues and eigenvectors in the input arrays, this function:
    1. Reconstructs the tensor from eigenvalues and eigenvectors
    2. Returns the lower triangular elements for all points

    Args:
        eigenvalues (numpy.ndarray): Array of shape (num_point, dim) containing
                                    eigenvalues for each point
        eigenvectors (numpy.ndarray): Array of shape (num_point, dim, dim) containing
                                     eigenvectors for each point

    Returns:
        numpy.ndarray: Array of shape (num_point, num_met) containing recomposed
                      metric tensors in lower triangular form.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If dimension is not 1, 2, or 3

    Examples:
        >>> # Recompose a field of 2D tensors
        >>> eigenvalues = np.array([
        ...     [3.5, 1.5],  # Eigenvalues for first tensor
        ...     [4.0, 1.0],  # Eigenvalues for second tensor
        ... ])
        >>> eigenvectors = np.array([
        ...     [[0.8, -0.6], [0.6, 0.8]],  # Eigenvectors for first tensor
        ...     [[1.0, 0.0], [0.0, 1.0]],   # Eigenvectors for second tensor
        ... ])
        >>> recomposed = recompose_metric_field(eigenvalues, eigenvectors)
        >>> print(recomposed.shape)  # (2, 3)

        >>> # Workflow: decompose then recompose should give original tensors
        >>> metrics = np.array([[2.0, 0.5, 1.0], [3.0, 0.0, 1.0]])
        >>> eigenvals, eigenvecs = decompose_metric_field(metrics)
        >>> reconstructed = recompose_metric_field(eigenvals, eigenvecs)
        >>> np.allclose(metrics, reconstructed)  # Should be True
    """
    return _metric.recompose_metric_field(
        np.asarray(eigenvalues, dtype=np.float64), np.asarray(eigenvectors, dtype=np.float64)
    )


def metric_edge_length_at_endpoints(edges: NDArray, coords: NDArray, metrics: NDArray) -> NDArray:
    r"""Calculate edge length in Riemannian metric space evaluated at endpoints.

    For each edge, computes the metric length ||e||_M at both endpoints,
    where e is the edge vector and M is the metric tensor.

    Args:
        edges (numpy.ndarray): Array of shape (num_edge, 2) containing node indices
                              for each edge.
        coords (numpy.ndarray): Array of shape (num_node, dim) containing node
                               coordinates.
        metrics (numpy.ndarray): Array of shape (num_node, num_met) containing
                                metric tensors at each node in lower triangular form.
                                num_met is 1, 3, or 6 for 1D, 2D, or 3D tensors.

    Returns:
        numpy.ndarray: Array of shape (num_edge, 2) containing metric lengths
                      at both endpoints of each edge.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements
        RuntimeError: If edge contains invalid node index

    Examples:
        >>> # Calculate metric edge lengths for 2D mesh
        >>> edges = np.array([
        ...     [0, 1],  # Edge from node 0 to node 1
        ...     [1, 2],  # Edge from node 1 to node 2
        ...     [0, 2],  # Edge from node 0 to node 2
        ... ])
        >>> coords = np.array([
        ...     [0.0, 0.0],  # Node 0 coordinates
        ...     [1.0, 0.0],  # Node 1 coordinates
        ...     [0.0, 1.0],  # Node 2 coordinates
        ... ])
        >>> metrics = np.array([
        ...     [2.0, 0.0, 1.0],  # Metric at node 0 [M00, M10, M11]
        ...     [1.0, 0.0, 2.0],  # Metric at node 1
        ...     [1.5, 0.1, 1.5],  # Metric at node 2
        ... ])
        >>> lengths = metric_edge_length_at_endpoints(edges, coords, metrics)
        >>> print(lengths.shape)  # (3, 2)

        >>> # For 3D case
        >>> coords_3d = np.array([
        ...     [0.0, 0.0, 0.0],
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ... ])
        >>> metrics_3d = np.array([
        ...     [2.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # 3D metric [M00, M10, M11, M20, M21, M22]
        ...     [1.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        ...     [1.5, 0.1, 1.5, 0.0, 0.0, 1.0],
        ... ])
        >>> lengths_3d = metric_edge_length_at_endpoints(edges, coords_3d, metrics_3d)
    """
    return _metric.metric_edge_length_at_endpoints(
        np.asarray(edges, dtype=np.int32), np.asarray(coords, dtype=np.float64), np.asarray(metrics, dtype=np.float64)
    )


def metric_edge_length(edges: NDArray, coords: NDArray, metrics: NDArray, eps: float = 1e-12) -> NDArray:
    r"""Calculate integrated edge length in Riemannian metric space.

    Uses geometric integration approach based on Alauzet, Loseille, et al. at Inria.
    For each edge, computes the integrated metric length using the formula:
    - If ratio r ≈ 1: l_M = l_a
    - Otherwise: l_M = l_a * (r - 1) / ln(r)
    where r = l_b / l_a and l_a, l_b are metric lengths at endpoints.

    Args:
        edges (numpy.ndarray): Array of shape (num_edge, 2) containing node indices
                              for each edge.
        coords (numpy.ndarray): Array of shape (num_node, dim) containing node
                               coordinates.
        metrics (numpy.ndarray): Array of shape (num_node, num_met) containing
                                metric tensors at each node in lower triangular form.
                                num_met is 1, 3, or 6 for 1D, 2D, or 3D tensors.
        eps (float): Tolerance for ratio comparison to determine when to use
                    geometric integration vs. simple average (default: 1e-12).

    Returns:
        numpy.ndarray: Array of shape (num_edge,) containing integrated metric
                      lengths for each edge.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements
        RuntimeError: If edge contains invalid node index

    Examples:
        >>> # Calculate integrated metric edge lengths for 2D mesh
        >>> edges = np.array([
        ...     [0, 1],  # Edge from node 0 to node 1
        ...     [1, 2],  # Edge from node 1 to node 2
        ...     [0, 2],  # Edge from node 0 to node 2
        ... ])
        >>> coords = np.array([
        ...     [0.0, 0.0],  # Node 0 coordinates
        ...     [1.0, 0.0],  # Node 1 coordinates
        ...     [0.0, 1.0],  # Node 2 coordinates
        ... ])
        >>> metrics = np.array([
        ...     [2.0, 0.0, 1.0],  # Metric at node 0 [M00, M10, M11]
        ...     [1.0, 0.0, 2.0],  # Metric at node 1
        ...     [1.5, 0.1, 1.5],  # Metric at node 2
        ... ])
        >>> integrated_lengths = metric_edge_length(edges, coords, metrics)
        >>> print(integrated_lengths.shape)  # (3,)

        >>> # Use custom tolerance for ratio comparison
        >>> lengths_custom = metric_edge_length(edges, coords, metrics, eps=1e-10)

        >>> # For anisotropic metrics, the integrated length differs from Euclidean
        >>> euclidean_lengths = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=1)
        >>> print("Euclidean lengths:", euclidean_lengths)
        >>> print("Metric lengths:", integrated_lengths)
    """
    return _metric.metric_edge_length(
        np.asarray(edges, dtype=np.int32),
        np.asarray(coords, dtype=np.float64),
        np.asarray(metrics, dtype=np.float64),
        eps,
    )
