#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
extern "C" {
#include <libmeshb7.h>
}

namespace py = pybind11;

PYBIND11_MODULE(libmeshb, m) {
    m.doc() = "Python bindings for libMeshb";

    // Expose GmfOpenMesh function
    m.def("GmfOpenMesh", [](const char *filename, int mode, int& version, int& dim) {
        return GmfOpenMesh(filename, mode, &version, &dim);
    }, py::arg("filename"), py::arg("mode"), py::arg("version"), py::arg("dim"),
       "Open a *.meshb file with the given filename, mode, and version");


    // Expose GmfCloseMesh function
    m.def("GmfCloseMesh", [](int mesh) {
        return GmfCloseMesh(mesh);
    }, py::arg("mesh"),
       "Close the mesh with the given mesh ID");
}