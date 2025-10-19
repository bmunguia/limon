from pathlib import Path

import numpy as np
import pytest
from limon.mesh import (
    load_mesh,
    write_mesh,
    write_solution,
    load_mesh_and_solution,
    write_mesh_and_solution,
)


@pytest.fixture
def meshpath_in_3d():
    """Path to example 3D GMF (.meshb) mesh."""
    return Path('libMeshb/sample_meshes/quad.meshb')


@pytest.fixture
def meshpath_in_2d():
    """Path to example 2D GMF (.meshb) mesh."""
    return Path('example/square/square.meshb')


@pytest.fixture
def labelpath(output_dir):
    """Path to solution label map file."""
    return output_dir / 'labels.dat'


@pytest.fixture
def mesh_data_3d(meshpath_in_3d):
    """Load the 3D mesh and create a sample solution."""
    data = load_mesh(meshpath_in_3d)
    coords, elements, boundaries = data

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
def mesh_data_2d(meshpath_in_2d):
    """Load the 2D mesh and create a sample solution."""
    data = load_mesh(meshpath_in_2d)
    coords, elements, boundaries = data

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


def test_load_meshb(mesh_data_3d):
    """Test reading a mesh."""
    coords, elements, boundaries, solution = mesh_data_3d

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(boundaries) > 0  # The sphere example only has a surface
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_write_meshb(mesh_data_3d, output_dir):
    """Test writing a mesh."""
    coords, elements, boundaries, _ = mesh_data_3d

    # Output paths
    meshpath_out = output_dir / 'sphere_with_sol.mesh'

    # Write the mesh with the solution
    write_mesh(meshpath_out, coords, elements, boundaries)

    # Assert that the files were created
    assert meshpath_out.exists()


def test_write_solb(mesh_data_3d, output_dir):
    """Test writing a mesh with a solution."""
    coords, _, _, solution = mesh_data_3d

    # Output paths
    solpath_out = output_dir / 'sphere_with_sol.solb'

    # Write the mesh with the solution
    num_point = coords.shape[0]
    dim = coords.shape[1]
    write_solution(solpath_out, solution, num_point, dim)

    # Assert that the files were created
    assert solpath_out.exists()


def test_write_meshb_and_solb(mesh_data_3d, output_dir):
    """Test writing a mesh with a solution."""
    coords, elements, boundaries, solution = mesh_data_3d

    # Output paths
    meshpath_out = output_dir / 'sphere_with_sol.mesh'
    solpath_out = output_dir / 'sphere_with_sol.solb'

    # Write the mesh with the solution
    write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        coords,
        elements,
        boundaries,
        solution,
    )

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_solution_keys_solb(mesh_data_2d, output_dir, labelpath):
    """Test that solution field keys are preserved when writing and reading back a mesh."""
    coords, elements, boundaries, original_solution = mesh_data_2d

    # Output paths
    meshpath_out = output_dir / 'square_with_fields.mesh'
    solpath_out = output_dir / 'square_with_fields.solb'

    # Write the mesh with the original solution
    write_success = write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        coords,
        elements,
        boundaries,
        original_solution,
    )
    print(f'Original solution written: {write_success}')

    # Assert that writing was successful
    assert write_success, 'Failed to write mesh with solution'
    assert meshpath_out.exists(), 'Output mesh file was not created'
    assert solpath_out.exists(), 'Output solution file was not created'
    assert labelpath.exists(), 'Output solution label map was not created'

    # Read the mesh and solution back
    _, _, _, reloaded_solution = load_mesh_and_solution(
        meshpath_out,
        solpath_out,
        labelpath=labelpath,
    )

    # Check that the solution was read correctly
    assert reloaded_solution is not None, 'Failed to read solution'

    # Check that all original solution keys are present in the reread solution
    original_keys = set(original_solution.keys())
    reread_keys = set(reloaded_solution.keys())

    print(f'Original solution keys: {original_keys}')
    print(f'Re-read solution keys: {reread_keys}')

    assert original_keys.issubset(reread_keys), 'Some solution fields were not preserved'

    # Check that the field types and sizes match
    for key in original_keys:
        assert key in reloaded_solution, f'Field "{key}" missing in reread solution'

        # Check shape
        assert reloaded_solution[key].shape == original_solution[key].shape, f'Shape mismatch for field "{key}"'

        # Check approximate values (allowing for small floating point differences)
        assert np.allclose(reloaded_solution[key], original_solution[key], rtol=1e-5), (
            f'Values differ for field "{key}"'
        )
