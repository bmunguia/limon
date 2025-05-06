from pybind11.setup_helpers import Pybind11Extension, build_ext

def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension(
            "libmeshb",
            ["pymeshb/libMeshb/libmeshb.cpp"],
            include_dirs=["libMeshb/sources"],
            libraries=["Meshb.7"],
            library_dirs=["libMeshb/sources"],
        ),
    ]

    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
        }
    )