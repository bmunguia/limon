/** @file refine.cpp
 *  @brief Python bindings for mesh refinement.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <pybind11/pybind11.h>
#include "include/refine_2d.hpp"
// #include "include/refine_3d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_refine, m) {
    m.doc() = "Python bindings for mesh refinement";

    m.def("refine_2d", &limon::mesh::refine_2d,
          py::arg("coords"),
          py::arg("elements"),
          py::arg("boundaries"),
          "Refine a 2D mesh by one level (uniform). "
          "Returns a dictionary with keys: coords, elements, boundaries, dim, num_point.");

    // m.def("refine_3d", &limon::mesh::refine_3d,
    //       py::arg("coords"),
    //       py::arg("elements"),
    //       py::arg("boundaries"),
    //       "Refine a 3D mesh by one level (uniform).");
}
