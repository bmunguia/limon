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

#include "include/mesh.hpp"
#include "include/solution.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libgmf, m) {
    m.doc() = "Python bindings for libMeshb mesh and solution files";

    /**
     * Read mesh data from a GMF mesh (.meshb) file.
     *
     * @param meshpath Path to the mesh file
     * @return Dictionary containing mesh data with keys: coords, elements, boundaries, dim, num_point
     */
    m.def("load_mesh", &limon::gmf::load_mesh,
          py::arg("meshpath"),
          "Read a meshb file and return mesh data as a dictionary.");

    /**
     * Write mesh data to a GMF mesh (.meshb) file.
     *
     * @param meshpath Path to the mesh file
     * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
     * @return Boolean indicating success
     */
    m.def("write_mesh", &limon::gmf::write_mesh,
          py::arg("meshpath"),
          py::arg("mesh_data"),
          "Write mesh data to a meshb file.");

    /**
     * Read solution data from a GMF solution (.solb) file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @param labelpath Path to the map between solution strings and ref IDs
     * @return Dictionary of solution fields
     */
    m.def("load_solution", &limon::gmf::load_solution,
          py::arg("solpath"),
          py::arg("num_ver"),
          py::arg("dim"),
          py::arg("labelpath") = "",
          "Read a solb solution file.");

    /**
     * Write solution data to a GMF solution (.solb) file.
     *
     * @param solpath Path to the solution file
     * @param sol_data Dictionary of solution fields
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @return Boolean indicating success
     */
    m.def("write_solution", &limon::gmf::write_solution,
          py::arg("solpath"),
          py::arg("sol_data"),
          py::arg("num_ver"),
          py::arg("dim"),
          "Write solution data to a solb solution file.");
}