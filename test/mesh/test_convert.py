from pathlib import Path

import pytest
from pymeshb.mesh import load_mesh_and_solution, write_mesh_and_solution


@pytest.fixture
def su2_meshpath_in():
    """Path to example SU2 (.su2) mesh."""
    return Path('example/naca0012/NACA0012_inv.su2')


@pytest.fixture
def su2_solpath_in():
    """Path to example SU2 (.dat) solution."""
    return Path('example/naca0012/restart_flow.dat')


@pytest.fixture
def gmf_meshpath_in():
    """Path to example GMF (.meshb) mesh."""
    return Path('example/square/square.mesh')


@pytest.fixture
def gmf_solpath_in():
    """Path to example GMF (.solb) solution."""
    return Path('example/square/square.solb')


@pytest.fixture
def markerpath_in():
    """Path to input marker map file."""
    return Path('example/square/markers.dat')


@pytest.fixture
def labelpath_in():
    """Path to input solution label map file."""
    return Path('example/square/labels.dat')


@pytest.fixture
def markerpath(output_dir):
    """Path to marker map file."""
    return output_dir / 'markers.dat'


@pytest.fixture
def labelpath(output_dir):
    """Path to solution label map file."""
    return output_dir / 'labels.dat'


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def su2_mesh_data(su2_meshpath_in, su2_solpath_in, markerpath, labelpath):
    """Load the 2D SU2 mesh and solution."""
    data = load_mesh_and_solution(
        su2_meshpath_in,
        su2_solpath_in,
        markerpath=markerpath,
        labelpath=labelpath,
    )
    return data


@pytest.fixture
def gmf_mesh_data(gmf_meshpath_in, gmf_solpath_in, markerpath_in, labelpath_in):
    """Load the 2D GMF mesh and binary solution."""
    data = load_mesh_and_solution(
        gmf_meshpath_in,
        gmf_solpath_in,
        labelpath=labelpath_in,
    )
    return data


def test_su2_to_gmf(su2_mesh_data, output_dir):
    """Test reading a SU2 mesh file and solution, and writing it to GMF files."""
    coords, elements, boundaries, solution = su2_mesh_data

    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.meshb'
    solpath_out = output_dir / 'naca_with_sol.solb'

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

    # # Assert that the new solution matches the example
    # assert solpath_out.read_bytes() == solpath_in_binary.read_bytes()


def test_gmf_to_su2(gmf_mesh_data, output_dir, markerpath_in):
    """Test reading a GMF mesh file and solution, and writing it to SU2 files."""
    coords, elements, boundaries, solution = gmf_mesh_data

    # Output paths
    meshpath_out = output_dir / 'square.su2'
    solpath_out = output_dir / 'square.csv'

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

    # # Assert that the new solution matches the example
    # assert solpath_out.read_bytes() == solpath_in_binary.read_bytes()
