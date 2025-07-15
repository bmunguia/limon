import numpy as np
import pytest
from pymeshb.metric import decompose, recompose


@pytest.mark.parametrize(
    'lower_tri, test_name',
    [
        (np.array([7.0]), '1D tensor'),
        (np.array([4.0, 1.0, 3.0]), '2D tensor'),
        (np.array([6.0, 2.0, 5.0, 1.0, 3.0, 4.0]), '3D tensor'),
    ],
    ids=['1D-tensor', '2D-tensor', '3D-tensor'],
)
def test_decompose_and_recompose(lower_tri, test_name):
    """Test decompose() and recompose() for symmetric tensors of different dimensions."""
    # Diagonalize the tensor
    eigenvalues, eigenvectors = decompose(lower_tri)

    # Assert eigenvalues are sorted in descending order
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), (
        f'Eigenvalues are not sorted in descending order for {test_name}'
    )

    # Recombine the tensor from eigenvalues and eigenvectors
    reconstructed = recompose(eigenvalues, eigenvectors)

    # Assert the reconstructed tensor matches the original tensor
    assert np.allclose(reconstructed, lower_tri, atol=1e-6), f'Reconstructed {test_name} does not match the original'
