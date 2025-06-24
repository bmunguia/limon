import numpy as np
from numpy.typing import NDArray

from . import _metric


def decompose(lower_tri: NDArray) -> tuple[NDArray, NDArray]:
    r"""
    Diagonalize a symmetric tensor represented by its lower triangular elements.

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
    r"""
    Recombine eigenvalues and eigenvectors to reconstruct the tensor.

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
    return _metric.recompose(
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.float64)
    )


def perturb(
    eigenvalues: NDArray,
    eigenvectors: NDArray,
    delta_eigenvals: NDArray,
    rotation_angles: NDArray
) -> tuple[NDArray, NDArray]:
    r"""
    Apply perturbations to eigenvalues and eigenvectors.

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
        np.asarray(rotation_angles, dtype=np.float64)
    )


def perturb_metric_field(
    metrics: NDArray,
    delta_eigenvals: NDArray,
    rotation_angles: NDArray
) -> NDArray:
    r"""
    Perturb a field of metric tensors.

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
        np.asarray(rotation_angles, dtype=np.float64)
    )

def integrate_metric_field(
    metrics: NDArray,
    volumes: NDArray,
    norm: float = 2.0,
) -> float:
    r"""
    Integrate the determinant of metric tensors over volumes.

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
        norm (float): Choice of norm for Lp-norm normalization.

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
        >>> integral = integrate_metric_field(metrics, volumes, norm=2.0)
        
        >>> # Integrate determinants of a field of 3D tensors
        >>> metrics_3d = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # 3D tensor [a00, a10, a11, a20, a21, a22]
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> volumes_3d = np.array([0.05, 0.08])
        >>> integral_3d = integrate_metric_field(metrics_3d, volumes_3d, norm=2.0)
    """
    return _metric.integrate_metric_field(
        np.asarray(metrics, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        norm,
    )

def normalize_metric_field(
    metrics: NDArray,
    norm: float,
    metric_integral: float,
    complexity: int,
) -> NDArray:
    r"""
    Normalize a field of metric tensors.

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
        norm (float): Choice of norm for Lp-norm normalization.
        metric_integral (float): Integral of metric determinants over the domain.
        complexity (int): Target complexity for normalization.

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
        >>> norm = 2.0
        >>> metric_integral = 15.5  # Previously computed integral
        >>> complexity = 1000.0    # Target complexity
        >>> normalized = normalize_metric_field(metrics, norm, metric_integral, complexity)
        
        >>> # Normalize a field of 3D tensors
        >>> metrics_3d = np.array([
        ...     [2.0, 0.5, 1.0, 0.1, 0.2, 3.0],  # 3D tensor
        ...     [3.0, 0.0, 1.0, 0.3, 0.1, 2.0],  # Second 3D tensor
        ... ])
        >>> normalized_3d = normalize_metric_field(metrics_3d, norm=2.0, metric_integral=25.8, 
                                                   complexity=5000)
    """
    return _metric.normalize_metric_field(
        np.asarray(metrics, dtype=np.float64),
        norm,
        metric_integral,
        complexity,
    )