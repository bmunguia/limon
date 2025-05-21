import pytest
from pathlib import Path

import pymeshb


@pytest.fixture
def mesh_data():
    '''Load the 2D mesh and create a sample solution.'''
    meshpath_in = 'example/square.su2'
    markerpath = Path('example/markers.dat')
    coords, elements, boundaries, solution = pymeshb.read_mesh(meshpath_in, markerpath=str(markerpath))

    return coords, elements, boundaries, markerpath, solution


@pytest.fixture
def output_dir(request):
    '''Create a persistent output directory for test files.'''
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_read_su2(mesh_data):
    '''Test reading a SU2 mesh file.'''
    coords, elements, boundaries, markerpath, solution = mesh_data

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert markerpath.exists()  # Ensure that the marker file exists
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_write_su2(mesh_data, output_dir):
    '''Test writing a SU2 mesh file and reading it back.'''
    coords, elements, boundaries, markerpath, solution = mesh_data

    # Output paths
    meshpath_out = output_dir / 'square_with_sol.su2'
    solpath_out = '' # str(output_dir / 'square_with_sol.dat'

    # Write the mesh with the solution
    pymeshb.write_mesh(str(meshpath_out), coords, elements, boundaries,
                       markerpath=str(markerpath), solpath=str(solpath_out),
                       solution=solution)

    # Assert that the files were created
    assert meshpath_out.exists()
    # assert solpath_out.exists()