from pathlib import Path

import numpy.testing as npt
import pytest
from limon.mesh import load_mesh_and_solution, write_mesh_and_solution


@pytest.fixture
def meshpath_in():
    """Path to example SU2 (.su2) mesh."""
    return Path('data/naca0012/NACA0012_inv.su2')


@pytest.fixture
def solpath_in_binary():
    """Path to example SU2 binary (.dat) solution."""
    return Path('data/naca0012/restart_flow.dat')


@pytest.fixture
def solpath_in_ascii():
    """Path to example SU2 ASCII (.csv) solution."""
    return Path('data/naca0012/restart_flow.csv')


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def mesh_data_binary(meshpath_in, solpath_in_binary):
    """Load the 2D mesh and binary solution."""
    return load_mesh_and_solution(meshpath_in, solpath_in_binary)


@pytest.fixture
def mesh_data_ascii(meshpath_in, solpath_in_ascii):
    """Load the 2D mesh and ASCII solution."""
    return load_mesh_and_solution(meshpath_in, solpath_in_ascii)


def test_read_su2_binary(mesh_data_binary):
    """Test reading a binary (.dat) SU2 mesh file."""
    coords = mesh_data_binary['coords']
    elements = mesh_data_binary['elements']
    boundaries = mesh_data_binary['boundaries']
    solution = mesh_data_binary['solution']

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert mesh_data_binary['markers']  # Marker names are returned in memory
    assert list(mesh_data_binary['labels'].values()) == list(solution)  # Field names follow the file order
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_read_su2_ascii(mesh_data_ascii):
    """Test reading an ASCII (.csv) SU2 mesh file."""
    coords = mesh_data_ascii['coords']
    elements = mesh_data_ascii['elements']
    boundaries = mesh_data_ascii['boundaries']
    solution = mesh_data_ascii['solution']

    # Assert that the mesh data is loaded correctly
    assert coords.shape[0] > 0  # Ensure there are points
    assert len(elements) > 0  # Ensure there are elements
    assert mesh_data_ascii['markers']  # Marker names are returned in memory
    assert list(mesh_data_ascii['labels'].values()) == list(solution)  # Field names follow the file order
    assert isinstance(solution, dict)  # Ensure solution is a dictionary


def test_su2_binary_to_binary(mesh_data_binary, output_dir, solpath_in_binary):
    """Test reading a SU2 mesh file and binary solution, and writing it back
    with a binary solution.
    """
    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.dat'

    # Write the mesh with the solution
    write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        mesh_data_binary,
    )

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_binary.read_bytes()


def test_su2_binary_to_ascii(mesh_data_binary, output_dir, solpath_in_ascii):
    """Test reading a SU2 mesh file and binary solution, and writing it back
    with an ASCII solution.
    """
    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.csv'

    # Write the mesh with the solution
    write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        mesh_data_binary,
    )

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_ascii.read_bytes()


def test_su2_ascii_to_ascii(mesh_data_ascii, output_dir, solpath_in_ascii):
    """Test reading a SU2 mesh file and ASCII solution, and writing it back
    with an ASCII solution.
    """
    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.csv'

    # Write the mesh with the solution
    write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        mesh_data_ascii,
    )

    # Assert that the files were created
    assert meshpath_out.exists()
    assert solpath_out.exists()

    # Assert that the new solution matches the example
    assert solpath_out.read_bytes() == solpath_in_ascii.read_bytes()


def test_su2_ascii_to_binary(mesh_data_ascii, output_dir, meshpath_in, solpath_in_binary):
    """Test reading a SU2 mesh file and ASCII solution, and writing it back
    with a binary solution.
    """
    # Output paths
    meshpath_out = output_dir / 'naca_with_sol.su2'
    solpath_out = output_dir / 'naca_with_sol.dat'

    # Write the mesh with the solution
    write_mesh_and_solution(
        meshpath_out,
        solpath_out,
        mesh_data_ascii,
    )

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
    mesh_data_in = load_mesh_and_solution(meshpath_in, solpath_in)
    sol_in = mesh_data_in['solution']

    # Read the rewritten solution
    mesh_data_out = load_mesh_and_solution(meshpath_out, solpath_out)
    sol_out = mesh_data_out['solution']

    assert sol_out.keys() == sol_in.keys(), 'Solution fields mismatch'

    for key in sol_in.keys():
        npt.assert_allclose(
            sol_out[key],
            sol_in[key],
            rtol=1e-6,
            atol=1e-8,
        )
