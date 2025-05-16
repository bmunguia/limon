import os
import subprocess
import shutil
from pathlib import Path

def build(setup_kwargs: dict | None = None):
    r"""Build the C++ extensions using CMake."""
    # Get conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX')

    # Create build directory
    build_dir = Path('build')
    build_dir.mkdir(exist_ok=True)

    # Run CMake
    cmd = [
        'cmake',
        '-S', '.',
        '-B', 'build',
        '-DCMAKE_BUILD_TYPE=Release'
    ]
    if conda_prefix:
        cmd.extend(['-DCMAKE_PREFIX_PATH=' + conda_prefix])

    subprocess.run(cmd, check=True)

    # Build
    subprocess.run(['cmake', '--build', 'build'], check=True)

    # Copy the libraries to the install location
    lib_names = [
        'libgmf',
        'libsu2',
        '_metric',
    ]
    lib_dirs = [
        Path('pymeshb/mesh/gmf'),
        Path('pymeshb/mesh/su2'),
        Path('pymeshb/metric'),
    ]
    for lib_name, lib_dir in zip(lib_names, lib_dirs):
        copy_lib(lib_name, lib_dir, build_dir)

    # Return empty dict if called by poetry
    return {} if setup_kwargs is not None else None

def copy_lib(lib_name: str, lib_dir: Path, build_dir: Path):
    r"""Copy the built libraries."""
    # Create necessary directory structure
    lib_dir.mkdir(exist_ok=True)

    # Copy the lib to the specified directory
    _lib = list(build_dir.glob(f'*{lib_name}.*'))

    if _lib:
        shutil.copy2(_lib[0], lib_dir)
    else:
        print(f'Warning: {lib_name} library not found in build dir')

if __name__ == '__main__':
    build()