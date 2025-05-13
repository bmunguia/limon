import numpy as np
import pytest
from pathlib import Path

import pymeshb
from pymeshb.metric import perturb_metric_field


def print_perturb_comparison(met, met_pert):
    """Print a comparison between the initial and perturbed metric."""
    met_init_str = 'Initial metric:   ['
    met_pert_str = 'Perturbed metric: ['
    for i in range(met.shape[1]):
        met_init_str += f'{met[0][i]:4.0f}'
        met_pert_str += f'{met_pert[0][i]:4.0f}'
        if i < len(met[0]) - 1:
            met_init_str += ', '
            met_pert_str += ', '

    print('\n' + met_init_str + ']')
    print(met_pert_str + ']')


@pytest.fixture
def mesh_data():
    """Load the 2D mesh and create a sample solution."""
    meshpath_in = "test/mesh/square.mesh"
    coords, elements, solution = pymeshb.read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        "Metric": np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
    }

    # Make sample metric diagonal
    solution["Metric"][:, 0] = 1e3
    solution["Metric"][:, 2] = 1e4

    return coords, elements, solution, num_point, num_dim


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path("output") / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_write_mesh_with_metric(mesh_data, output_dir):
    """Test writing a 2D mesh with a metric."""
    coords, elements, solution, _, _ = mesh_data

    # Output paths
    meshpath_out = output_dir / "square_with_met.meshb"
    solpath_out = output_dir / "square_with_met.solb"

    # Write the mesh with the solution
    pymeshb.write_mesh(str(meshpath_out), coords, elements,
                       solpath=str(solpath_out), solution=solution)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_perturb_eigenvalues(mesh_data, output_dir):
    """Test perturbing only the eigenvalues of the 2D metric field."""
    coords, elements, solution, num_point, num_dim = mesh_data

    assert num_dim == 2, "This test is specifically for 2D meshes"

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    delta_eigenvals[:, 0] = -np.log(10)
    delta_eigenvals[:, 1] = np.log(10)

    # Use zero rotation angles to isolate eigenvalue effects
    rotation_angles = np.zeros((num_point, 1))

    # Perturb the metric field (eigenvalues only)
    perturbed_metrics_eig = perturb_metric_field(
        solution["Metric"],
        delta_eigenvals,
        rotation_angles
    )

    print_perturb_comparison(solution["Metric"], perturbed_metrics_eig)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics_eig.shape == solution["Metric"].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics_eig, solution["Metric"])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        "Metric": perturbed_metrics_eig
    }

    # Output paths
    pert_meshpath_out = output_dir / "square_with_eig_pert_only.meshb"
    pert_solpath_out = output_dir / "square_with_eig_pert_only.solb"

    # Write the mesh with perturbed metrics
    pymeshb.write_mesh(str(pert_meshpath_out), coords, elements,
                      solpath=str(pert_solpath_out), solution=perturbed_solution)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_orientation(mesh_data, output_dir):
    """Test perturbing only the orientation of the 2D metric field."""
    coords, elements, solution, num_point, num_dim = mesh_data

    assert num_dim == 2, "This test is specifically for 2D meshes"

    # Use zero perturbations to isolate rotation effects
    delta_eigenvals = np.zeros((num_point, num_dim))

    # Create rotation angles - for 2D we only need 1 angle
    rotation_angles = np.zeros((num_point, 1))
    rotation_angles[:, 0] = np.pi / 2

    # Perturb the metric field (rotation only)
    perturbed_metrics_rot = perturb_metric_field(
        solution["Metric"],
        delta_eigenvals,
        rotation_angles
    )

    print_perturb_comparison(solution["Metric"], perturbed_metrics_rot)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics_rot.shape == solution["Metric"].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics_rot, solution["Metric"])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        "Metric": perturbed_metrics_rot
    }

    # Output paths
    pert_meshpath_out = output_dir / "square_with_rot_pert_only.meshb"
    pert_solpath_out = output_dir / "square_with_rot_pert_only.solb"

    # Write the mesh with perturbed metrics
    pymeshb.write_mesh(str(pert_meshpath_out), coords, elements,
                      solpath=str(pert_solpath_out), solution=perturbed_solution)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_metric_field(mesh_data, output_dir):
    """Test perturbing the 2D metric field."""
    coords, elements, solution, num_point, num_dim = mesh_data

    assert num_dim == 2, "This test is specifically for 2D meshes"

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    delta_eigenvals[:, 0] = -np.log(20)
    delta_eigenvals[:, 1] = np.log(2)

    # Create perturbation arrays for eigenvector orientations
    rotation_angles = np.zeros((num_point, 1))
    rotation_angles[:, 0] = np.pi / 6

    # Perturb the metric field (both eigenvalues and rotation)
    perturbed_metrics = perturb_metric_field(
        solution["Metric"],
        delta_eigenvals,
        rotation_angles
    )

    print_perturb_comparison(solution["Metric"], perturbed_metrics)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics.shape == solution["Metric"].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics, solution["Metric"])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        "Metric": perturbed_metrics
    }

    # Output paths
    pert_meshpath_out = output_dir / "square_with_combined_pert.meshb"
    pert_solpath_out = output_dir / "square_with_combined_pert.solb"

    # Write the mesh with perturbed metrics
    pymeshb.write_mesh(str(pert_meshpath_out), coords, elements,
                       solpath=str(pert_solpath_out), solution=perturbed_solution)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()