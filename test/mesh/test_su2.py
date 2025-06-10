from pathlib import Path

import numpy.testing as npt
import pytest
from pymeshb.mesh import read_mesh, write_mesh


@pytest.fixture
def meshpath_in():
    """Path to example SU2 (.su2) mesh."""
    return Path('example/naca0012/NACA0012_inv.su2')


@pytest.fixture
def solpath_in_binary():
    """Path to example SU2 binary (.dat) solution."""
    return Path('example/naca0012/restart_flow.dat')


@pytest.fixture
def solpath_in_ascii():
    """Path to example SU2 ASCII (.csv) solution."""
    return Path('example/naca0012/restart_flow.csv')


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
def mesh_data_binary(meshpath_in, solpath_in_binary, markerpath, labelpath):
    """Load the 2D mesh and binary solution."""
    data = read_mesh(str(meshpath_in), markerpath=str(markerpath),
                     solpath=str(solpath_in_binary), labelpath=str(labelpath),
                     read_sol=True)
    return data


@pytest.fixture
def mesh_data_ascii(meshpath_in, solpath_in_ascii, markerpath, labelpath):
    """Load the 2D mesh and ASCII solution."""
    data = read_mesh(str(meshpath_in), markerpath=str(markerpath),
                     solpath=str(solpath_in_ascii), labelpath=str(labelpath),
                     read_sol=True)
    return data


def test_read_su2_binary(mesh_data_binary, markerpath, labelpath):
    """Test reading a binary (.dat) SU2 mesh file."""
    coords, elements, boundaries, solution = mesh_data_binary

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert markerpath.exists()  # Ensure that the marker file exists
    assert labelpath.exists()  # Ensure that the solution label file exists
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_read_su2_ascii(mesh_data_ascii, markerpath, labelpath):
    """Test reading an ASCII (.csv) SU2 mesh file."""
    coords, elements, boundaries, solution = mesh_data_ascii

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert markerpath.exists()  # Ensure that the marker file exists
    assert labelpath.exists()  # Ensure that the solution label file exists
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_su2_binary_to_binary(mesh_data_binary, output_dir, markerpath,
                              labelpath, solpath_in_binary):
    """Test reading a SU2 mesh file and binary solution, and writing it back
    with a binary solution.
    """
    coords, elements, boundaries, solution = mesh_data_binary

    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.dat'

    # Write the mesh with the solution
    write_mesh(str(meshpath_out), coords, elements, boundaries,
               markerpath=str(markerpath), solpath=str(solpath_out),
               solution=solution, write_sol=True)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_binary.read_bytes()


def test_su2_binary_to_ascii(mesh_data_binary, output_dir, markerpath,
                             labelpath, solpath_in_ascii):
    """Test reading a SU2 mesh file and binary solution, and writing it back
    with an ASCII solution.
    """
    coords, elements, boundaries, solution = mesh_data_binary

    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.csv'

    # Write the mesh with the solution
    write_mesh(str(meshpath_out), coords, elements, boundaries,
               markerpath=str(markerpath), solpath=str(solpath_out),
               labelpath=str(labelpath), solution=solution,
               write_sol=True)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_ascii.read_bytes()


def test_su2_ascii_to_ascii(mesh_data_ascii, output_dir, markerpath,
                            labelpath, solpath_in_ascii):
    """Test reading a SU2 mesh file and ASCII solution, and writing it back
    with an ASCII solution.
    """
    coords, elements, boundaries, solution = mesh_data_ascii

    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.csv'

    # Write the mesh with the solution
    write_mesh(str(meshpath_out), coords, elements, boundaries,
               markerpath=str(markerpath), solpath=str(solpath_out),
               labelpath=str(labelpath), solution=solution,
               write_sol=True)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_ascii.read_bytes()


def test_su2_ascii_to_binary(mesh_data_ascii, output_dir, markerpath,
                             labelpath, meshpath_in, solpath_in_binary):
    """Test reading a SU2 mesh file and ASCII solution, and writing it back
    with a binary solution.
    """
    coords, elements, boundaries, solution = mesh_data_ascii

    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.dat'

    # Write the mesh with the solution
    write_mesh(str(meshpath_out), coords, elements, boundaries,
               markerpath=str(markerpath), solpath=str(solpath_out),
               labelpath=str(labelpath), solution=solution,
               write_sol=True)

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    # NOTE: using a separate test to check that solutions are close, because the CSV
    #       doesn't have the same precision as the binary, so the binary from the conversion
    #       won't have the same bytes as the binary output by SU2
    compare_solutions(meshpath_in, solpath_in_binary, meshpath_out, solpath_out)


def compare_solutions(meshpath_in, solpath_in, meshpath_out, solpath_out):
    """Read the new solution and compare it to the original."""
    # Read the original solution
    _, _, _, sol_in = read_mesh(str(meshpath_in), solpath=str(solpath_in),
                                read_sol=True)

    # Read the rewritten solution
    _, _, _, sol_out = read_mesh(str(meshpath_out), solpath=str(solpath_out),
                                 read_sol=True)

    assert sol_out.keys() == sol_in.keys(), 'Solution fields mismatch'

    for key in sol_in.keys():
        npt.assert_allclose(
            sol_out[key],
            sol_in[key],
            rtol=1e-6,
            atol=1e-8,
        )
