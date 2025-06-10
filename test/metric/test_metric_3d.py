from pathlib import Path

import numpy as np
import pytest
from pymeshb.mesh import read_mesh, write_mesh
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
    """Load the 3D mesh and create a sample solution."""
    meshpath_in = 'libMeshb/sample_meshes/quad.meshb'
    coords, elements, boundaries = read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Create a sample solution dictionary
    solution = {
        'Metric': np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
    }

    # Make sample metric diagonal
    solution['Metric'][:, 0] = 1e1
    solution['Metric'][:, 2] = 1e2
    solution['Metric'][:, 5] = 1e3

    return coords, elements, boundaries, solution, num_point, num_dim


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_write_mesh_with_metric(mesh_data, output_dir):
    """Test writing a 3D mesh with a metric."""
    coords, elements, boundaries, solution, _, _ = mesh_data

    # Output paths
    meshpath_out = output_dir / 'sphere_with_met.meshb'
    solpath_out = output_dir / 'sphere_with_met.solb'

    # Write the mesh with the solution
    write_mesh(str(meshpath_out), coords, elements, boundaries,
               solpath=str(solpath_out), solution=solution,
               write_sol=True)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_perturb_eigenvalues(mesh_data, output_dir):
    """Test perturbing only the eigenvalues of the 3D metric field."""
    coords, elements, boundaries, solution, num_point, num_dim = mesh_data

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    delta_eigenvals[:, 0] = -np.log(100)
    delta_eigenvals[:, 1] = np.log(10)
    delta_eigenvals[:, 2] = np.log(10)

    # Use zero rotation angles to isolate eigenvalue effects
    num_angle = 1 if num_dim == 2 else 3
    rotation_angles = np.zeros((num_point, num_angle))

    # Perturb the metric field (eigenvalues only)
    perturbed_metrics_eig = perturb_metric_field(
        solution['Metric'],
        delta_eigenvals,
        rotation_angles,
    )

    print_perturb_comparison(solution['Metric'], perturbed_metrics_eig)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics_eig.shape == solution['Metric'].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics_eig, solution['Metric'])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        'Metric': perturbed_metrics_eig
    }

    # Output paths
    pert_meshpath_out = output_dir / 'sphere_with_eig_pert_only.meshb'
    pert_solpath_out = output_dir / 'sphere_with_eig_pert_only.solb'

    # Write the mesh with perturbed metrics
    write_mesh(str(pert_meshpath_out), coords, elements, boundaries,
               solpath=str(pert_solpath_out), solution=perturbed_solution,
               write_sol=True)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_orientation(mesh_data, output_dir):
    """Test perturbing only the orientation of the 3D metric field."""
    coords, elements, boundaries, solution, num_point, num_dim = mesh_data

    # Use zero perturbations to isolate rotation effects
    delta_eigenvals = np.zeros((num_point, num_dim))

    # Create perturbation arrays for eigenvector orientations
    num_angle = 1 if num_dim == 2 else 3
    rotation_angles = np.zeros((num_point, num_angle))
    rotation_angles[:, 0] = np.pi / 2
    rotation_angles[:, 1] = np.pi / 2

    # Perturb the metric field (rotation only)
    perturbed_metrics_rot = perturb_metric_field(
        solution['Metric'],
        delta_eigenvals,
        rotation_angles,
    )

    print_perturb_comparison(solution['Metric'], perturbed_metrics_rot)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics_rot.shape == solution['Metric'].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics_rot, solution['Metric'])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        'Metric': perturbed_metrics_rot
    }

    # Output paths
    pert_meshpath_out = output_dir / 'sphere_with_rot_pert_only.meshb'
    pert_solpath_out = output_dir / 'sphere_with_rot_pert_only.solb'

    # Write the mesh with perturbed metrics
    write_mesh(str(pert_meshpath_out), coords, elements, boundaries,
               solpath=str(pert_solpath_out), solution=perturbed_solution,
               write_sol=True)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_metric_field(mesh_data, output_dir):
    """Test perturbing the 3D metric field."""
    coords, elements, boundaries, solution, num_point, num_dim = mesh_data

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    delta_eigenvals[:, 0] = -np.log(5)
    delta_eigenvals[:, 1] = np.log(5)
    delta_eigenvals[:, 2] = np.log(2)

    # Create perturbation arrays for eigenvector orientations
    num_angle = 1 if num_dim == 2 else 3
    rotation_angles = np.zeros((num_point, num_angle))
    # Add some small rotation angles (in radians)
    rotation_angles[:, 0] = np.pi / 6
    rotation_angles[:, 1] = np.pi / 4
    rotation_angles[:, 2] = 0.1

    # Perturb the metric field
    perturbed_metrics = perturb_metric_field(
        solution['Metric'],
        delta_eigenvals,
        rotation_angles,
    )

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics.shape == solution['Metric'].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics, solution['Metric'])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {
        'Metric': perturbed_metrics
    }

    # Output paths
    pert_meshpath_out = output_dir / 'sphere_with_combined_pert_met.meshb'
    pert_solpath_out = output_dir / 'sphere_with_combined_pert_met.solb'

    # Write the mesh with perturbed metrics
    write_mesh(str(pert_meshpath_out), coords, elements, boundaries,
               solpath=str(pert_solpath_out), solution=perturbed_solution,
               write_sol=True)

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()