from pathlib import Path

import pytest
from limon.mesh import LimonIOError, load_mesh, load_mesh_and_solution, write_mesh, write_mesh_and_solution


@pytest.fixture
def su2_meshpath_in():
    """Path to example SU2 (.su2) mesh."""
    return Path('data/naca0012/NACA0012_inv.su2')


@pytest.fixture
def su2_solpath_in():
    """Path to example SU2 (.dat) solution."""
    return Path('data/naca0012/restart_flow.dat')


@pytest.fixture
def gmf_meshpath_in():
    """Path to example GMF (.meshb) mesh."""
    return Path('data/square/square.mesh')


@pytest.fixture
def gmf_solpath_in():
    """Path to example GMF (.solb) solution."""
    return Path('data/square/square.solb')


@pytest.fixture
def output_dir(request):
    """Create a persistent output directory for test files."""
    out_dir = Path('output') / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def su2_mesh_data(su2_meshpath_in, su2_solpath_in):
    """Load the 2D SU2 mesh and solution."""
    return load_mesh_and_solution(su2_meshpath_in, su2_solpath_in)


@pytest.fixture
def gmf_mesh_data(gmf_meshpath_in, gmf_solpath_in):
    """Load the 2D GMF mesh and binary solution."""
    return load_mesh_and_solution(gmf_meshpath_in, gmf_solpath_in)


def test_su2_to_gmf(su2_mesh_data, output_dir):
    """Test reading a SU2 mesh file and solution, and writing it to GMF files."""
    meshpath_out = output_dir / 'naca_with_sol.meshb'
    solpath_out = output_dir / 'naca_with_sol.solb'

    write_mesh_and_solution(meshpath_out, solpath_out, su2_mesh_data)

    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_gmf_to_su2(gmf_mesh_data, output_dir):
    """Test reading a GMF mesh file and solution, and writing it to SU2 files."""
    meshpath_out = output_dir / 'square.su2'
    solpath_out = output_dir / 'square.csv'

    write_mesh_and_solution(meshpath_out, solpath_out, gmf_mesh_data)

    assert meshpath_out.exists()
    assert solpath_out.exists()


def test_names_survive_su2_gmf_su2_round_trip(su2_mesh_data, output_dir):
    """Marker and field names travel through GMF files with no caller-held dictionaries."""
    gmf_mesh, gmf_sol = output_dir / 'naca.meshb', output_dir / 'naca.solb'
    write_mesh_and_solution(gmf_mesh, gmf_sol, su2_mesh_data)

    back = load_mesh_and_solution(gmf_mesh, gmf_sol)
    assert back['markers'] == su2_mesh_data['markers']
    assert list(back['solution']) == list(su2_mesh_data['solution'])

    su2_mesh, su2_sol = output_dir / 'naca_again.su2', output_dir / 'naca_again.csv'
    write_mesh_and_solution(su2_mesh, su2_sol, back)
    again = load_mesh_and_solution(su2_mesh, su2_sol)
    assert again['markers'] == su2_mesh_data['markers']
    assert list(again['solution']) == list(su2_mesh_data['solution'])


def test_gmf_names_default_without_sidecar(su2_mesh_data, output_dir):
    """Without a sidecar or caller names, GMF fields are REF_<n>; explicit names override the sidecar."""
    gmf_mesh, gmf_sol = output_dir / 'naca.meshb', output_dir / 'naca.solb'
    write_mesh_and_solution(gmf_mesh, gmf_sol, su2_mesh_data, write_names=False)

    bare = load_mesh_and_solution(gmf_mesh, gmf_sol)
    assert bare['markers'] == {}
    assert list(bare['solution']) == [f'REF_{i + 1}' for i in range(len(su2_mesh_data['solution']))]

    names = list(su2_mesh_data['solution'])
    named = load_mesh_and_solution(gmf_mesh, gmf_sol, marker_map=su2_mesh_data['markers'], names=names)
    assert named['markers'] == su2_mesh_data['markers']
    assert list(named['solution']) == names


def test_inputs_are_not_mutated(su2_mesh_data, output_dir):
    marker_map = {1: 'wall'}
    gmf_mesh = output_dir / 'naca.meshb'
    write_mesh(gmf_mesh, su2_mesh_data, write_names=False)
    load_mesh(gmf_mesh, marker_map=marker_map)
    assert marker_map == {1: 'wall'}


def test_errors_raise(output_dir):
    with pytest.raises(LimonIOError):
        load_mesh(output_dir / 'missing.meshb')
    with pytest.raises(LimonIOError):
        load_mesh(output_dir / 'mesh.unknown')
    with pytest.raises(LimonIOError):
        write_mesh(output_dir / 'bad.su2', {})
