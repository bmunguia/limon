#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/surface_metric.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_geometry, m) {
    m.doc() = "Mesh geometry kernels";

    m.def("surface_metric", &limon::geometry::surface_metric,
          py::arg("mesh_data"),
          py::arg("geodev"),
          py::arg("mode") = "angle",
          py::arg("hmin") = 1e-8,
          py::arg("hmax") = 1e8,
          "Surface-geometry metric of a 2-D mesh as an (N, 3) array of [xx, xy, yy].");
}
