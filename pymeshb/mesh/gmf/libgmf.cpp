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

    /**
      * Read mesh data from an SU2 file.
      *
      * @param meshpath Path to the mesh file
      * @param solpath Path to the solution file (optional)
      * @param read_sol Whether to read solution data (default: false)
      * @return Tuple containing coordinates, elements, and optionally solution data
      */
    m.def("read_mesh", &pymeshb::gmf::read_mesh,
          py::arg("meshpath"), py::arg("solpath") = "", py::arg("read_sol") = false,
          "Read a meshb file and return nodes, elements, and optionally solution data.");

    /**
      * Write mesh data to an SU2 mesh file.
      *
      * @param meshpath Path to the mesh file
      * @param coords Coordinates of each node
      * @param elements Dictionary of mesh elements
      * @param solpath Path to the solution file (optional)
      * @param sol Dictionary of solution data (optional)
      * @return Boolean indicating success
      */
    m.def("write_mesh", &pymeshb::gmf::write_mesh,
          py::arg("meshpath"), py::arg("coords"), py::arg("elements"),
          py::arg("solpath") = "", py::arg("sol") = py::dict(),
          "Write nodes, elements, and optionally solution data to a meshb file.");

    /**
     * Read solution data from an SU2 solution file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @return Dictionary of solution fields
     */
    m.def("read_solution", &pymeshb::gmf::read_solution,
          py::arg("solpath"), py::arg("num_ver"), py::arg("dim"),
          "Read a meshb solution file.");

    /**
   * Write solution data to an SU2 solution file.
   *
   * @param solpath Path to the solution file
   * @param sol Dictionary of solution fields
   * @param num_ver Number of vertices
   * @param dim Mesh dimension
   * @return Boolean indicating success
   */
    m.def("write_solution", &pymeshb::gmf::write_solution,
          py::arg("solpath"), py::arg("sol_data"), py::arg("num_ver"),
          py::arg("dim"), py::arg("version"),
          "Write solution data to a meshb solution file.");
}