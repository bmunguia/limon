/** @file libgmf.cpp
 *  @brief Python bindings for libMeshb library.
 *
 *  Implementation of Python bindings for the libMeshb library, providing
 *  functionality to read and write meshb files, including scalar, vector,
 *  and tensor fields in solutions.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mesh.hpp"
#include "solution.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libgmf, m) {
    m.doc() = "Python bindings for libMeshb";

    m.def("read_mesh", &pymeshb::gmf::read_mesh,
          py::arg("meshpath"), py::arg("solpath") = "", py::arg("read_sol") = false,
          "Read a meshb file and return nodes, elements, and optionally solution data.");

    m.def("write_mesh", &pymeshb::gmf::write_mesh,
          py::arg("meshpath"), py::arg("coords"), py::arg("elements"),
          py::arg("solpath") = "", py::arg("sol") = py::dict(),
          "Write nodes, elements, and optionally solution data to a meshb file.");

    m.def("read_solution", &pymeshb::gmf::read_solution,
          py::arg("solpath"), py::arg("num_ver"), py::arg("dim"),
          "Read a meshb solution file.");

    m.def("write_solution", &pymeshb::gmf::write_solution,
          py::arg("solpath"), py::arg("sol_data"), py::arg("num_ver"),
          py::arg("dim"), py::arg("version"),
          "Write solution data to a meshb solution file.");
}