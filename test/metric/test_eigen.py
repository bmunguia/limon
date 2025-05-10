import numpy as np
import pytest

from pymeshb.metric import decompose, recompose


def test_decompose_and_recompose_2d():
    """Test decompose() and recompose() for a 2D symmetric tensor."""
    # Define a 2D symmetric tensor in lower triangular form
    # [a00, a10, a11]
    lower_tri = np.array([4.0, 1.0, 3.0])

    # Diagonalize the tensor
    eigenvalues, eigenvectors = decompose(lower_tri)

    # Assert eigenvalues are sorted in descending order
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), "Eigenvalues are not sorted in descending order"

    # Recombine the tensor from eigenvalues and eigenvectors
    reconstructed = recompose(eigenvalues, eigenvectors)

    # Assert the reconstructed tensor matches the original tensor
    assert np.allclose(reconstructed, lower_tri, atol=1e-6), "Reconstructed tensor does not match the original"


def test_decompose_and_recompose_3d():
    """Test decompose() and recompose() for a 3D symmetric tensor."""
    # Define a 3D symmetric tensor in lower triangular form
    # [a00, a10, a11, a20, a21, a22]
    lower_tri = np.array([6.0, 2.0, 5.0, 1.0, 3.0, 4.0])

    # Diagonalize the tensor
    eigenvalues, eigenvectors = decompose(lower_tri)

    # Assert eigenvalues are sorted in descending order
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), "Eigenvalues are not sorted in descending order"

    # Recombine the tensor from eigenvalues and eigenvectors
    reconstructed = recompose(eigenvalues, eigenvectors)

    # Assert the reconstructed tensor matches the original tensor
    assert np.allclose(reconstructed, lower_tri, atol=1e-6), "Reconstructed tensor does not match the original"


def test_decompose_and_recompose_1d():
    """Test decompose() and recompose() for a 1D symmetric tensor."""
    # Define a 1D symmetric tensor in lower triangular form
    # [a00]
    lower_tri = np.array([7.0])

    # Diagonalize the tensor
    eigenvalues, eigenvectors = decompose(lower_tri)

    # Assert eigenvalues are correct
    assert np.allclose(eigenvalues, [7.0]), "Eigenvalues are incorrect for 1D tensor"

    # Recombine the tensor from eigenvalues and eigenvectors
    reconstructed = recompose(eigenvalues, eigenvectors)

    # Assert the reconstructed tensor matches the original tensor
    assert np.allclose(reconstructed, lower_tri, atol=1e-6), "Reconstructed tensor does not match the original"