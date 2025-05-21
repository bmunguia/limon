import numpy as np
import pytest
from pathlib import Path

import pymeshb


@pytest.fixture
def mesh_data_3d():
    """Load the 3D mesh and create a sample solution."""
    meshpath_in = 'libMeshb/sample_meshes/quad.meshb'
    coords, elements, boundaries, solution = pymeshb.read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        'Temperature': coords[:, 0] + 10,
        'Velocity': np.zeros((num_point, num_dim)),
        'Metric': np.zeros((num_point, (num_dim * (num_dim + 1)) // 2)),
    }
    solution['Velocity'][:, 1] = coords[:, 1] * 2

    # Make sample metric diagonal
    solution['Metric'][:, 0] = 1e1
    solution['Metric'][:, 2] = 1e2
    solution['Metric'][:, 5] = 1e3

    return coords, elements, boundaries, solution


@pytest.fixture
def mesh_data_2d():
    """Load the 2D mesh and create a sample solution."""
    meshpath_in = 'example/square/square.meshb'
    coords, elements, boundaries, solution = pymeshb.read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        'Temperature': np.linspace(293.0, 373.0, num_point),
        'Velocity': np.zeros((num_point, num_dim)),
        'Metric': np.zeros((num_point, (num_dim * (num_dim + 1)) // 2)),
        'Pressure': 101325.0 + 10000.0 * coords[:, 0],
        'GradientT': np.zeros((num_point, num_dim)),
    }

    # Add some meaningful values
    solution['Velocity'][:, 0] = coords[:, 0] * 10.0
    solution['Velocity'][:, 1] = coords[:, 1] * 5.0

    # Make sample metric diagonal
    solution['Metric'][:, 0] = 1e1  # m11
    solution['Metric'][:, 2] = 1e2  # m22

    # Gradient field varies with position
    solution['GradientT'][:, 0] = 5.0 * (coords[:, 0] + 0.1)  # dT/dx
    solution['GradientT'][:, 1] = 8.0 * (coords[:, 1] + 0.1)  # dT/dy

    return coords, elements, boundaries, solution


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_read_meshb(mesh_data_3d):
    """Test reading a mesh."""
    coords, elements, boundaries, solution = mesh_data_3d

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(boundaries) > 0  # The sphere example only has a surface
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_write_solb(mesh_data_3d, output_dir):
    """Test writing a mesh with a solution."""
    coords, elements, boundaries, solution = mesh_data_3d

    # Output paths
    meshpath_out = output_dir / 'sphere_with_sol.mesh'
    solpath_out = output_dir / 'sphere_with_sol.sol'

    # Write the mesh with the solution
    pymeshb.write_mesh(str(meshpath_out), coords, elements, boundaries,
                       solpath=str(solpath_out), solution=solution)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_solution_keys_solb(mesh_data_2d, output_dir):
    """Test that solution field keys are preserved when writing and reading back a mesh."""
    coords, elements, boundaries, original_solution = mesh_data_2d

    # Output paths
    meshpath_out = output_dir / 'square_with_fields.mesh'
    solpath_out = output_dir / 'square_with_fields.sol'

    # Write the mesh with the original solution
    write_success = pymeshb.write_mesh(str(meshpath_out), coords, elements, boundaries,
                                       solpath=str(solpath_out), solution=original_solution)
    print(f'Original solution written: {write_success}')

    # Assert that writing was successful
    assert write_success, 'Failed to write mesh with solution'
    assert meshpath_out.exists(), 'Output mesh file was not created'
    assert solpath_out.exists(), 'Output solution file was not created'

    # Read the mesh and solution back
    _, _, _, reread_solution = pymeshb.read_mesh(
        str(meshpath_out),
        solpath=str(solpath_out),
        read_sol=True
    )

    # Check that the solution was read correctly
    assert reread_solution is not None, 'Failed to read solution'

    # Check that all original solution keys are present in the reread solution
    original_keys = set(original_solution.keys())
    reread_keys = set(reread_solution.keys())

    print(f'Original solution keys: {original_keys}')
    print(f'Re-read solution keys: {reread_keys}')

    assert original_keys.issubset(reread_keys), 'Some solution fields were not preserved'

    # Check that the field types and sizes match
    for key in original_keys:
        assert key in reread_solution, f'Field "{key}" missing in reread solution'

        # Check shape
        assert reread_solution[key].shape == original_solution[key].shape, \
            f'Shape mismatch for field "{key}"'

        # Check approximate values (allowing for small floating point differences)
        assert np.allclose(reread_solution[key], original_solution[key], rtol=1e-5), \
            f'Values differ for field "{key}"'

    # Test reading specific fields
    # Read only Temperature field
    _, _, _, partial_solution = pymeshb.read_mesh(
        str(meshpath_out),
        solpath=str(solpath_out),
        read_sol=True
    )

    # Verify that the partial reading worked correctly
    assert 'Temperature' in partial_solution, 'Failed to read Temperature field'
    assert np.allclose(partial_solution['Temperature'], original_solution['Temperature'], rtol=1e-5), \
        'Temperature values differ in partial read'

    # Additional validation - write a subset of fields and verify they're preserved
    subset_solution = {
        'Temperature': original_solution['Temperature'],
        'Metric': original_solution['Metric']
    }

    # Output paths for subset
    subset_meshpath_out = output_dir / 'square_subset_fields.meshb'
    subset_solpath_out = output_dir / 'square_subset_fields.solb'

    # Write the mesh with subset of fields
    pymeshb.write_mesh(
        str(subset_meshpath_out),
        coords,
        elements,
        boundaries,
        solpath=str(subset_solpath_out),
        solution=subset_solution
    )

    # Read back the subset
    _, _, _, subset_reread = pymeshb.read_mesh(
        str(subset_meshpath_out),
        solpath=str(subset_solpath_out),
        read_sol=True
    )

    # Check that only the subset keys are present
    subset_keys = set(subset_solution.keys())
    subset_reread_keys = set(subset_reread.keys())

    print(f'Subset solution keys: {subset_keys}')
    print(f'Re-read subset keys: {subset_reread_keys}')

    assert subset_keys.issubset(subset_reread_keys), 'Some subset fields were not preserved'
    assert len(subset_reread_keys) == len(subset_keys), 'Extra fields found in subset read'

    # Final verification that we can read fields with correct types
    for key in subset_reread_keys:
        if subset_reread[key].ndim == 1:
            print(f'Field "{key}" is a scalar field')
        elif subset_reread[key].ndim == 2 and subset_reread[key].shape[1] == 2:
            print(f'Field "{key}" is a vector field')
        elif subset_reread[key].ndim == 2 and subset_reread[key].shape[1] == 3:
            print(f'Field "{key}" is a symmetric tensor field')