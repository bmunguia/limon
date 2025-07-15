from pathlib import Path

import numpy as np
import pytest
from pymeshb.mesh import read_mesh, write_mesh
from pymeshb.metric import perturb_metric_field, metric_edge_length, metric_edge_length_at_endpoints


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
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(
    params=[
        {
            'dim': 2,
            'meshpath': 'example/square/square.mesh',
            'file_prefix': 'square',
            'metric_values': {'diag_indices': [0, 2], 'values': [1e3, 1e4]},
        },
        {
            'dim': 3,
            'meshpath': 'example/cube/cube.mesh',
            'file_prefix': 'cube',
            'metric_values': {'diag_indices': [0, 2, 5], 'values': [1e1, 1e2, 1e3]},
        },
    ],
    ids=['2D-mesh', '3D-mesh'],
)
def mesh_data(request):
    """Load mesh and create sample solution for different dimensions."""
    config = request.param
    coords, elements, boundaries = read_mesh(config['meshpath'])

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    # Verify dimension matches expectation
    assert num_dim == config['dim'], f'Expected {config["dim"]}D mesh, got {num_dim}D'

    # Create a sample solution dictionary
    solution = {'Metric': np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))}

    # Set diagonal metric values
    for idx, value in zip(config['metric_values']['diag_indices'], config['metric_values']['values']):
        solution['Metric'][:, idx] = value

    return coords, elements, boundaries, solution, num_point, num_dim, config


def test_write_mesh_with_metric(mesh_data, output_dir):
    """Test writing mesh with a metric for different dimensions."""
    coords, elements, boundaries, solution, _, _, config = mesh_data

    # Output paths
    meshpath_out = output_dir / f'{config["file_prefix"]}_with_met.meshb'
    solpath_out = output_dir / f'{config["file_prefix"]}_with_met.solb'

    # Write the mesh with the solution
    write_mesh(
        str(meshpath_out), coords, elements, boundaries, solpath=str(solpath_out), solution=solution, write_sol=True
    )

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_perturb_eigenvalues(mesh_data, output_dir):
    """Test perturbing only the eigenvalues of the metric field."""
    coords, elements, boundaries, solution, num_point, num_dim, config = mesh_data

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    if num_dim == 2:
        delta_eigenvals[:, 0] = -np.log(10)
        delta_eigenvals[:, 1] = np.log(10)
    else:  # 3D
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
    perturbed_solution = {'Metric': perturbed_metrics_eig}

    # Output paths
    pert_meshpath_out = output_dir / f'{config["file_prefix"]}_with_eig_pert_only.meshb'
    pert_solpath_out = output_dir / f'{config["file_prefix"]}_with_eig_pert_only.solb'

    # Write the mesh with perturbed metrics
    write_mesh(
        str(pert_meshpath_out),
        coords,
        elements,
        boundaries,
        solpath=str(pert_solpath_out),
        solution=perturbed_solution,
        write_sol=True,
    )

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_orientation(mesh_data, output_dir):
    """Test perturbing only the orientation of the metric field."""
    coords, elements, boundaries, solution, num_point, num_dim, config = mesh_data

    # Use zero perturbations to isolate rotation effects
    delta_eigenvals = np.zeros((num_point, num_dim))

    # Create rotation angles
    num_angle = 1 if num_dim == 2 else 3
    rotation_angles = np.zeros((num_point, num_angle))
    rotation_angles[:, 0] = np.pi / 2
    if num_dim == 3:
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
    perturbed_solution = {'Metric': perturbed_metrics_rot}

    # Output paths
    pert_meshpath_out = output_dir / f'{config["file_prefix"]}_with_rot_pert_only.meshb'
    pert_solpath_out = output_dir / f'{config["file_prefix"]}_with_rot_pert_only.solb'

    # Write the mesh with perturbed metrics
    write_mesh(
        str(pert_meshpath_out),
        coords,
        elements,
        boundaries,
        solpath=str(pert_solpath_out),
        solution=perturbed_solution,
        write_sol=True,
    )

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_perturb_metric_field(mesh_data, output_dir):
    """Test perturbing the metric field (both eigenvalues and orientation)."""
    coords, elements, boundaries, solution, num_point, num_dim, config = mesh_data

    # Create perturbation arrays for eigenvalues (in log space)
    delta_eigenvals = np.zeros((num_point, num_dim))
    if num_dim == 2:
        delta_eigenvals[:, 0] = -np.log(20)
        delta_eigenvals[:, 1] = np.log(2)
    else:  # 3D
        delta_eigenvals[:, 0] = -np.log(5)
        delta_eigenvals[:, 1] = np.log(5)
        delta_eigenvals[:, 2] = np.log(2)

    # Create perturbation arrays for eigenvector orientations
    num_angle = 1 if num_dim == 2 else 3
    rotation_angles = np.zeros((num_point, num_angle))
    rotation_angles[:, 0] = np.pi / 6
    if num_dim == 3:
        rotation_angles[:, 1] = np.pi / 4
        rotation_angles[:, 2] = 0.1

    # Perturb the metric field (both eigenvalues and rotation)
    perturbed_metrics = perturb_metric_field(
        solution['Metric'],
        delta_eigenvals,
        rotation_angles,
    )

    print_perturb_comparison(solution['Metric'], perturbed_metrics)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics.shape == solution['Metric'].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics, solution['Metric'])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {'Metric': perturbed_metrics}

    # Output paths
    pert_meshpath_out = output_dir / f'{config["file_prefix"]}_with_combined_pert.meshb'
    pert_solpath_out = output_dir / f'{config["file_prefix"]}_with_combined_pert.solb'

    # Write the mesh with perturbed metrics
    write_mesh(
        str(pert_meshpath_out),
        coords,
        elements,
        boundaries,
        solpath=str(pert_solpath_out),
        solution=perturbed_solution,
        write_sol=True,
    )

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


@pytest.fixture
def mesh_data_2d():
    """Load the 2D mesh and create a sample solution - specific for nonuniform test."""
    meshpath_in = 'example/square/square.mesh'
    coords, elements, boundaries = read_mesh(meshpath_in)

    num_point = coords.shape[0]
    num_dim = coords.shape[1]

    assert num_dim == 2, 'This fixture is specifically for 2D meshes'

    # Create a sample solution dictionary
    solution = {'Metric': np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))}

    # Make sample metric diagonal
    solution['Metric'][:, 0] = 1e3
    solution['Metric'][:, 2] = 1e4

    return coords, elements, boundaries, solution, num_point, num_dim


def test_nonuniform_perturb_metric_field(mesh_data_2d, output_dir):
    """Test nonuniformly perturbing the 2D metric field."""
    coords, elements, boundaries, solution, num_point, num_dim = mesh_data_2d

    # Create perturbation arrays for eigenvalues (in log space)
    log_max_old = np.log(max(solution['Metric'][0, [0, 2]]))
    log_min_old = np.log(min(solution['Metric'][0, [0, 2]]))
    r2 = pow(coords - 0.5, 2).sum(-1).clip(1e-2)
    log_max_new = np.log(1e3 / np.sqrt(r2))
    delta_eigenvals = np.zeros((num_point, num_dim))
    delta_eigenvals[:, 0] = log_max_new - log_max_old
    delta_eigenvals[:, 1] = np.log(1e2) - log_min_old

    # Create perturbation arrays for eigenvector orientations
    rotation_angles = np.zeros((num_point, 1))
    th_new = np.atan2(coords[:, 1] - 0.5, coords[:, 0] - 0.5)
    rotation_angles[:, 0] = th_new

    # Perturb the metric field (both eigenvalues and rotation)
    perturbed_metrics = perturb_metric_field(
        solution['Metric'],
        delta_eigenvals,
        rotation_angles,
    )

    print_perturb_comparison(solution['Metric'], perturbed_metrics)

    # Assert that the perturbed metrics have the same shape as the original
    assert perturbed_metrics.shape == solution['Metric'].shape

    # Assert that the perturbed metrics are different from the original
    assert not np.allclose(perturbed_metrics, solution['Metric'])

    # Create a new solution dictionary for perturbed metrics
    perturbed_solution = {'Metric': perturbed_metrics}

    # Output paths
    pert_meshpath_out = output_dir / 'square_with_nonuniform_pert.meshb'
    pert_solpath_out = output_dir / 'square_with_nonuniform_pert.solb'

    # Write the mesh with perturbed metrics
    write_mesh(
        str(pert_meshpath_out),
        coords,
        elements,
        boundaries,
        solpath=str(pert_solpath_out),
        solution=perturbed_solution,
        write_sol=True,
    )

    # Assert that the files were created
    assert pert_meshpath_out.exists()
    assert pert_solpath_out.exists()


def test_edge_length(mesh_data):
    """Test the computation of edge lengths for metric fields in both 2D and 3D."""
    coords, elements, boundaries, solution, num_point, num_dim, config = mesh_data

    # Extract edges from elements (first 4 elements, first 2 vertices of each)
    print(f'Available element types: {elements.keys()}')

    # Get the primary element type based on dimension
    if num_dim == 2:
        element_key = 'Triangles'
    else:  # 3D
        element_key = 'Hexahedra'

    edges = elements[element_key][:4, :2]
    print(f'Using edges from {element_key}: {edges}')

    # Compute edge lengths using the metric field
    edge_lengths = metric_edge_length(edges, coords, solution['Metric'])

    # Assert that the edge lengths have the expected shape
    assert edge_lengths.shape[0] == edges.shape[0], (
        f'Expected {edges.shape[0]} edge lengths, got {edge_lengths.shape[0]}'
    )
    assert edge_lengths.ndim == 1, f'Expected 1D array, got {edge_lengths.ndim}D'

    # Assert edge lengths are positive
    assert np.all(edge_lengths > 0), f'All edge lengths should be positive, got {edge_lengths}'

    # Optionally, print the edge lengths for inspection
    print(f'Computed edge lengths for {num_dim}D mesh: {edge_lengths}')


def test_edge_length_at_endpoints(mesh_data):
    """Test the computation of edge lengths at the endpoints for metric fields in both 2D and 3D."""
    coords, elements, boundaries, solution, num_point, num_dim, config = mesh_data

    # Extract edges from elements (first 4 elements, first 2 vertices of each)
    print(f'Available element types: {elements.keys()}')

    # Get the primary element type based on dimension
    if num_dim == 2:
        element_key = 'Triangles'
    else:  # 3D
        element_key = 'Hexahedra'

    edges = elements[element_key][:4, :2]
    print(f'Using edges from {element_key}: {edges}')

    # Compute edge lengths at endpoints using the metric field
    edge_lengths_endpoints = metric_edge_length_at_endpoints(edges, coords, solution['Metric'])

    # Assert that the edge lengths at endpoints have the expected shape
    expected_shape = (edges.shape[0], 2)
    assert edge_lengths_endpoints.shape == expected_shape, (
        f'Expected shape {expected_shape}, got {edge_lengths_endpoints.shape}'
    )
    assert edge_lengths_endpoints.ndim == 2, f'Expected 2D array, got {edge_lengths_endpoints.ndim}D'

    # Assert edge lengths are positive
    assert np.all(edge_lengths_endpoints > 0), f'All edge lengths should be positive, got {edge_lengths_endpoints}'

    # Optionally, print the edge lengths at endpoints for inspection
    print(f'Computed edge lengths at endpoints for {num_dim}D mesh: {edge_lengths_endpoints}')


@pytest.mark.parametrize(
    'dim, coords, edges, identity_metric, expected_lengths',
    [
        (
            2,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.array([1.0, 0.0, 1.0]),  # 2D identity metric (lower triangular)
            np.array([1.0, 1.0, np.sqrt(2)]),
        ),
        (
            3,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=np.int32),
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),  # 3D identity metric (lower triangular)
            np.array([1.0, 1.0, 1.0, np.sqrt(2)]),
        ),
    ],
    ids=['2D', '3D'],
)
def test_identity_metric_edge_length(dim, coords, edges, identity_metric, expected_lengths):
    """Test that identity metric gives Euclidean edge lengths."""
    num_points = coords.shape[0]
    metrics = np.tile(identity_metric, (num_points, 1))  # Same metric at all points

    # Test both functions
    lengths_endpoints = metric_edge_length_at_endpoints(edges, coords, metrics)
    lengths_integrated = metric_edge_length(edges, coords, metrics)

    # metric_edge_length_at_endpoints returns lengths at both endpoints (2D array)
    # For identity metric, both endpoints should give same length
    expected_shape = (edges.shape[0], 2)
    assert lengths_endpoints.shape == expected_shape, f'Expected shape {expected_shape}, got {lengths_endpoints.shape}'
    assert np.allclose(lengths_endpoints[:, 0], expected_lengths, atol=1e-12), (
        f'{dim}D endpoints start: expected {expected_lengths}, got {lengths_endpoints[:, 0]}'
    )
    assert np.allclose(lengths_endpoints[:, 1], expected_lengths, atol=1e-12), (
        f'{dim}D endpoints end: expected {expected_lengths}, got {lengths_endpoints[:, 1]}'
    )

    # metric_edge_length returns integrated length (1D array)
    assert np.allclose(lengths_integrated, expected_lengths, atol=1e-6), (
        f'{dim}D integrated: expected {expected_lengths}, got {lengths_integrated}'
    )


@pytest.mark.parametrize(
    'dim, coords, edges, metric, expected_scale_factor',
    [
        (
            2,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.array([4.0, 0.0, 4.0]),  # Scaled 2D identity metric
            2.0,  # sqrt(4.0)
        ),
        (
            3,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=np.int32),
            np.array([9.0, 0.0, 9.0, 0.0, 0.0, 9.0]),  # Scaled 3D identity metric
            3.0,  # sqrt(9.0)
        ),
    ],
    ids=['2D', '3D'],
)
def test_metric_edge_length(dim, coords, edges, metric, expected_scale_factor):
    """Test metric edge length functions with scaled metrics."""
    num_points = coords.shape[0]
    metrics = np.tile(metric, (num_points, 1))  # Same metric at all points

    # Expected Euclidean lengths for unit edges
    if dim == 2:
        euclidean_lengths = np.array([1.0, 1.0, np.sqrt(2)])
    else:  # 3D
        euclidean_lengths = np.array([1.0, 1.0, 1.0, np.sqrt(2)])

    expected_lengths = euclidean_lengths * expected_scale_factor

    # Test both functions
    lengths_endpoints = metric_edge_length_at_endpoints(edges, coords, metrics)
    lengths_integrated = metric_edge_length(edges, coords, metrics)

    # metric_edge_length_at_endpoints returns lengths at both endpoints (2D array)
    expected_shape = (edges.shape[0], 2)
    assert lengths_endpoints.shape == expected_shape, f'Expected shape {expected_shape}, got {lengths_endpoints.shape}'
    assert np.allclose(lengths_endpoints[:, 0], expected_lengths, atol=1e-12), (
        f'{dim}D scaled start: expected {expected_lengths}, got {lengths_endpoints[:, 0]}'
    )
    assert np.allclose(lengths_endpoints[:, 1], expected_lengths, atol=1e-12), (
        f'{dim}D scaled end: expected {expected_lengths}, got {lengths_endpoints[:, 1]}'
    )

    # metric_edge_length returns integrated length (1D array)
    assert np.allclose(lengths_integrated, expected_lengths, atol=1e-6), (
        f'{dim}D integrated: expected {expected_lengths}, got {lengths_integrated}'
    )
