import numpy as np
import pytest
from pathlib import Path

import pymeshb


@pytest.fixture
def mesh_data():
    """Load the mesh and create a sample solution."""
    meshpath_in = "libMeshb/sample_meshes/quad.meshb"
    coords, elements, solution = pymeshb.read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        "Temperature": coords[:, 0] + 10,
        "Velocity": np.zeros((num_point, num_dim)),
        "Metric": np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
    }
    solution["Velocity"][:, 1] = coords[:, 1] * 2

    # Make sample metric diagonal
    solution["Metric"][:, 0] = 1e1
    solution["Metric"][:, 2] = 1e2
    solution["Metric"][:, 5] = 1e3

    return coords, elements, solution


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path("output") / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_read_mesh(mesh_data):
    """Test reading a mesh."""
    coords, elements, solution = mesh_data

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_write_mesh_with_solution(mesh_data, output_dir):
    """Test writing a mesh with a solution."""
    coords, elements, solution = mesh_data

    # Output paths
    meshpath_out = output_dir / "quad_with_sol.meshb"
    solpath_out = output_dir / "quad_with_sol.solb"

    # Write the mesh with the solution
    pymeshb.write_mesh(str(meshpath_out), coords, elements,
                       solpath=str(solpath_out), solution=solution)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()