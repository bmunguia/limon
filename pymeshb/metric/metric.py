import numpy as np
from numpy.typing import NDArray

from pymeshb.metric._metric import decompose as _decompose
from pymeshb.metric._metric import recompose as _recompose
from pymeshb.metric._metric import perturb as _perturb
from pymeshb.metric._metric import perturb_metric_field as _perturb_metric_field


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
    return _decompose(np.asarray(lower_tri, dtype=np.float64))


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
    return _recompose(
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.float64)
    )


def perturb(
    eigenvalues: NDArray,
    eigenvectors: NDArray,
    val_perturbations: NDArray,
    vec_perturbations: NDArray
) -> tuple[NDArray, NDArray]:
    r"""
    Apply perturbations to eigenvalues and eigenvectors.

    This function applies perturbations to eigenvalues and eigenvectors and
    returns the perturbed eigenpairs. The perturbed eigenvectors are
    orthonormalized using the Gram-Schmidt process.

    Args:
        eigenvalues (numpy.ndarray): Original eigenvalues
        eigenvectors (numpy.ndarray): Original eigenvectors as columns
        val_perturbations (numpy.ndarray): Perturbation values for eigenvalues
        vec_perturbations (numpy.ndarray): Perturbation values for eigenvectors

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - perturbed_eigenvalues: Array of perturbed eigenvalues
            - perturbed_eigenvectors: Array of perturbed eigenvectors as columns

    Raises:
        RuntimeError: If dimensions of inputs don't match
        RuntimeError: If dimension is not 1, 2, or 3

    Examples:
        >>> # Apply perturbations to eigenvalues and eigenvectors
        >>> eigenvalues = np.array([4.0, 1.0])
        >>> eigenvectors = np.array([[0.8944, -0.4472], [0.4472, 0.8944]])
        >>> eig_pert = np.array([0.1, -0.05])
        >>> vec_pert = np.array([[0.01, 0.02], [-0.01, 0.01]])
        >>> pert_vals, pert_vecs = perturb(eigenvalues, eigenvectors, eig_pert, vec_pert)
    """
    return _perturb(
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.float64),
        np.asarray(val_perturbations, dtype=np.float64),
        np.asarray(vec_perturbations, dtype=np.float64)
    )


def perturb_metric_field(
    metrics: NDArray,
    val_perturbations: NDArray,
    vec_perturbations: NDArray
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
        val_perturbations (numpy.ndarray): Array of shape (num_point, dim) containing
 *                                        eigenvalue perturbations for each point.
                                          dim is 1, 2, or 3 for 1D, 2D, or 3D tensors.
        vec_perturbations (numpy.ndarray): Array of shape (num_point, dim, dim) containing
                                          eigenvector perturbations for each point.

    Returns:
        numpy.ndarray: Array of perturbed metric tensors in lower triangular form.

    Raises:
        RuntimeError: If input arrays have incompatible shapes
        RuntimeError: If tensor size is not 1, 3, or 6 elements

    Examples:
        >>> # Process a field of 2D tensors with perturbations
        >>> metrics = np.array([
        ...     [2.0, 0.5, 1.0],  # First tensor: [a00, a10, a11]
        ...     [3.0, 0.0, 1.0],  # Second tensor: [a00, a10, a11]
        ... ])
        >>> val_perturbations = np.array([
        ...     [0.1, -0.1],  # Log-space perturbations for first tensor's eigenvalues
        ...     [0.2, -0.05]  # Log-space perturbations for second tensor's eigenvalues
        ... ])
        >>> vec_perturbations = np.array([
        ...     [[0.01, 0.02], [-0.01, 0.01]],  # For first tensor
        ...     [[0.03, 0.01], [-0.02, 0.02]]   # For second tensor
        ... ])
        >>> perturbed_metrics = perturb_metric_field(metrics, val_perturbations, vec_perturbations)
    """
    return _perturb_metric_field(
        np.asarray(metrics, dtype=np.float64),
        np.asarray(val_perturbations, dtype=np.float64),
        np.asarray(vec_perturbations, dtype=np.float64)
    )