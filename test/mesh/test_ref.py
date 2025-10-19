from pathlib import Path
import tempfile
import pytest
from limon.mesh import load_ref_map, write_ref_map, get_ref_name, RefMapKind


@pytest.fixture
def expected_markers():
    """Expected marker reference map data."""
    return {1: 'lower', 2: 'right', 3: 'upper', 4: 'left'}


@pytest.fixture
def expected_labels():
    """Expected solution label reference map data."""
    return {1: 'Temperature', 2: 'Velocity', 3: 'Metric', 4: 'Pressure', 5: 'GradientT'}


@pytest.mark.parametrize('file_format', ['csv', 'dat', 'json'])
def test_load_markers_ref_map(file_format, expected_markers):
    """Test loading marker reference maps from different formats."""
    filename = f'example/square/markers.{file_format}'
    ref_map = load_ref_map(filename)

    assert ref_map == expected_markers
    assert len(ref_map) == 4


@pytest.mark.parametrize('file_format', ['csv', 'dat', 'json'])
def test_load_labels_ref_map(file_format, expected_labels):
    """Test loading solution label reference maps from different formats."""
    filename = f'example/square/labels.{file_format}'
    ref_map = load_ref_map(filename)

    assert ref_map == expected_labels
    assert len(ref_map) == 5


@pytest.mark.parametrize('file_format', ['csv', 'dat', 'json'])
def test_save_and_load_markers_roundtrip(file_format, expected_markers):
    """Test saving markers reference map and loading it back."""
    with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        # Save the reference map
        write_ref_map(expected_markers, tmp_filename, RefMapKind.Marker)

        # Load it back
        loaded_ref_map = load_ref_map(tmp_filename)
        assert loaded_ref_map == expected_markers
        assert len(loaded_ref_map) == 4

    finally:
        # Clean up
        Path(tmp_filename).unlink(missing_ok=True)


@pytest.mark.parametrize('file_format', ['csv', 'dat', 'json'])
def test_save_and_load_labels_roundtrip(file_format, expected_labels):
    """Test saving solution labels reference map and loading it back."""
    with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        # Save the reference map
        write_ref_map(expected_labels, tmp_filename, RefMapKind.Solution)

        # Load it back
        loaded_ref_map = load_ref_map(tmp_filename)  # Compare
        assert loaded_ref_map == expected_labels
        assert len(loaded_ref_map) == 5

    finally:
        # Clean up
        Path(tmp_filename).unlink(missing_ok=True)


def test_get_ref_name(expected_markers):
    """Test getting reference names from map."""
    # Test existing keys
    assert get_ref_name(expected_markers, 1) == 'lower'
    assert get_ref_name(expected_markers, 2) == 'right'
    assert get_ref_name(expected_markers, 3) == 'upper'
    assert get_ref_name(expected_markers, 4) == 'left'

    # Test non-existing key (should return default format)
    assert get_ref_name(expected_markers, 99) == 'REF_99'


def test_load_nonexistent_file():
    """Test loading a non-existent file returns empty map."""
    ref_map = load_ref_map('nonexistent_file.csv')
    assert ref_map == {}


def test_load_unsupported_extension():
    """Test loading file with unsupported extension."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b'1,test\n2,test2\n')
        tmp_filename = tmp_file.name

    try:
        # Should return empty map for unsupported extension
        ref_map = load_ref_map(tmp_filename)
        assert ref_map == {}

    finally:
        Path(tmp_filename).unlink(missing_ok=True)


def test_save_unsupported_extension(expected_markers):
    """Test saving to file with unsupported extension raises error."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        with pytest.raises(RuntimeError, match='Unsupported file extension'):
            write_ref_map(expected_markers, tmp_filename, RefMapKind.Marker)

    finally:
        Path(tmp_filename).unlink(missing_ok=True)


@pytest.mark.parametrize('ref_map_kind', [RefMapKind.Marker, RefMapKind.Solution])
def test_save_dat_file_headers(expected_markers, ref_map_kind):
    """Test that *.dat files are saved with correct headers based on RefMapKind."""
    with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        write_ref_map(expected_markers, tmp_filename, ref_map_kind)

        # Read the file and check headers
        with open(tmp_filename, 'r') as f:
            content = f.read()

        expected_kind_str = 'Solution' if ref_map_kind == RefMapKind.Solution else 'Marker'
        assert f'TITLE = "{expected_kind_str} ref map"' in content
        assert f'ZONE T= "{expected_kind_str} tags"' in content

    finally:
        Path(tmp_filename).unlink(missing_ok=True)


def test_empty_ref_map_roundtrip():
    """Test saving and loading empty reference map."""
    empty_map = {}

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        write_ref_map(empty_map, tmp_filename, RefMapKind.Marker)
        loaded_map = load_ref_map(tmp_filename)

        assert loaded_map == empty_map
        assert len(loaded_map) == 0

    finally:
        Path(tmp_filename).unlink(missing_ok=True)
