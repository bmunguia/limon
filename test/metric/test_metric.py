import numpy as np
import pytest
from pathlib import Path

import pymeshb
from pymeshb.metric import perturb_metric_field


@pytest.fixture
def mesh_data():
    """Load the mesh and create a sample solution."""
    meshpath_in = "libMeshb/sample_meshes/quad.meshb"
    coords, elements, solution = pymeshb.read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        "Metric": np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
    }

    # Make sample metric diagonal
    solution["Metric"][:, 0] = 1e1
    solution["Metric"][:, 2] = 1e2
    solution["Metric"][:, 5] = 1e3

    return coords, elements, solution, num_point, num_dim


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path("output") / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_write_mesh_with_metric(mesh_data, output_dir):
    """Test writing a mesh with a metric."""
    coords, elements, solution, _, _ = mesh_data

    # Output paths
    meshpath_out = output_dir / "quad_with_met.meshb"
    solpath_out = output_dir / "quad_with_met.solb"

    # Write the mesh with the solution
    pymeshb.write_mesh(str(meshpath_out), coords, elements,
                       solpath=str(solpath_out), solution=solution)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_perturb_metric_field(mesh_data, output_dir):
    """Test perturbing the metric field."""
    coords, elements, solution, num_point, num_dim = mesh_data

    # Create perturbation arrays for eigenvalues (in log space)
    val_perturbations = np.zeros((num_point, 3))
    val_perturbations[:, 0] = -np.log(100)
    val_perturbations[:, 1] = np.log(10)
    val_perturbations[:, 2] = np.log(10)

    # Create perturbation arrays for eigenvectors
    vec_perturbations = np.zeros((num_point, num_dim, num_dim))

    # Perturb the metric field
    perturbed_metrics = perturb_metric_field(
        solution["Metric"],
        val_perturbations,
        vec_perturbations
    )

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics.shape == solution["Metric"].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics, solution["Metric"])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        "Metric": perturbed_metrics
    }

    # Output paths
    pert_meshpath_out = output_dir / "quad_with_pert_met.meshb"
    pert_solpath_out = output_dir / "quad_with_pert_met.solb"

    # Write the mesh with perturbed metrics
    pymeshb.write_mesh(str(pert_meshpath_out), coords, elements,
                       solpath=str(pert_solpath_out), solution=perturbed_solution)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()