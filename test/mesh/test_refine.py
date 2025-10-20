from pathlib import Path

import pytest
from limon.mesh import load_mesh, write_mesh, refine_mesh


@pytest.fixture
def meshpath_in_2d():
    """Path to example 2D SU2 (.su2) mesh."""
    return Path('example/square/square.su2')


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def mesh_data_2d(meshpath_in_2d):
    """Load the 2D mesh."""
    mesh_data = load_mesh(meshpath_in_2d)
    return mesh_data


def test_refine_2d(mesh_data_2d, output_dir):
    """Test 2D mesh refinement."""
    # Get original mesh statistics
    original_coords = mesh_data_2d['coords']
    original_elements = mesh_data_2d['elements']
    original_boundaries = mesh_data_2d['boundaries']

    original_num_points = original_coords.shape[0]

    # Count original elements
    original_num_elements = 0
    for elem_type, elem_array in original_elements.items():
        original_num_elements += elem_array.shape[0]

    # Count original boundary elements
    original_num_boundary = 0
    for bound_type, bound_array in original_boundaries.items():
        original_num_boundary += bound_array.shape[0]

    print(
        f'Original mesh: {original_num_points} points, {original_num_elements} elements, {original_num_boundary} boundary elements'
    )

    # Refine the mesh
    refined_mesh_data = refine_mesh(mesh_data_2d)

    # Get refined mesh statistics
    refined_coords = refined_mesh_data['coords']
    refined_elements = refined_mesh_data['elements']
    refined_boundaries = refined_mesh_data['boundaries']

    refined_num_points = refined_coords.shape[0]

    # Count refined elements
    refined_num_elements = 0
    for elem_type, elem_array in refined_elements.items():
        refined_num_elements += elem_array.shape[0]

    # Count refined boundary elements
    refined_num_boundary = 0
    for bound_type, bound_array in refined_boundaries.items():
        refined_num_boundary += bound_array.shape[0]

    print(
        f'Refined mesh: {refined_num_points} points, {refined_num_elements} elements, {refined_num_boundary} boundary elements'
    )

    # Verify refinement results
    # For uniform refinement:
    # - Each triangle becomes 4 triangles, each quad becomes 4 quads
    # - Each edge becomes 2 edges
    # - New points = original_points + num_edges (one midpoint per edge)

    # Check that we have more elements (should be 4x for uniform refinement)
    assert refined_num_elements == 4 * original_num_elements, (
        f'Expected {4 * original_num_elements} elements, got {refined_num_elements}'
    )

    # Check that we have more boundary elements (should be 2x for edges)
    assert refined_num_boundary == 2 * original_num_boundary, (
        f'Expected {2 * original_num_boundary} boundary elements, got {refined_num_boundary}'
    )

    # Check that we have more points
    assert refined_num_points > original_num_points, (
        f'Expected more points than {original_num_points}, got {refined_num_points}'
    )

    # Write the refined mesh to output directory for inspection
    meshpath_out = output_dir / 'square_refined.mesh'
    write_mesh(meshpath_out, refined_mesh_data)

    # Assert that the file was created
    assert meshpath_out.exists(), 'Refined mesh file was not created'

    print(f'Refined mesh written to: {meshpath_out}')
