import subprocess
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext

def build_external(source_dir, build_dir):
    os.makedirs(build_dir, exist_ok=True)

    # Configure CMake
    cmake_configure_command = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DCMAKE_C_FLAGS=-fPIC",
        "-DCMAKE_CXX_FLAGS=-fPIC",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    try:
        subprocess.run(cmake_configure_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CMake configure failed: {e}")
        return

    # Build the project
    cmake_build_command = ["cmake", "--build", build_dir]
    try:
        subprocess.run(cmake_build_command, check=True)
    except:
        raise

def build(setup_kwargs):
    build_external("libMeshb", "build")

    ext_modules = [
        Pybind11Extension(
            "pymeshb.gamma.libmeshb",
            ["pymeshb/gamma/libmeshb.cpp"],
            include_dirs=["libMeshb/sources"],
            libraries=["Meshb.7"],
            library_dirs=["build/sources"],
            extra_compile_args=["-std=c++17"],
        ),
        Pybind11Extension(
            "pymeshb.metric._metric",
            ["pymeshb/metric/metric.cpp"],
            include_dirs=[],
            extra_compile_args=["-std=c++17"],
        ),
    ]

    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
        }
    )