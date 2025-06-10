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
    m.doc() = "Python bindings for libMeshb mesh and solution files";

    /**
     * Read mesh data from a GMF mesh (.meshb) file.
     *
     * @param meshpath Path to the mesh file
     * @return Tuple containing coordinates and element data
     */
    m.def("read_mesh", &pymeshb::gmf::read_mesh,
          py::arg("meshpath"),
          "Read a meshb file and return coordinates and element data.");

    /**
     * Write mesh data to a GMF mesh (.meshb) file.
     *
     * @param meshpath Path to the mesh file
     * @param coords Coordinates of each node
     * @param elements Dictionary of mesh elements
     * @param boundaries Dictionary of mesh boundary elements
     * @param solpath Path to the solution file (optional)
     * @param labelpath Path to the map between solution strings and ref IDs (optional)
     * @param sol Dictionary of solution data (optional)
     * @return Boolean indicating success
     */
    m.def("write_mesh", &pymeshb::gmf::write_mesh,
          py::arg("meshpath"), 
          py::arg("coords"), 
          py::arg("elements"),
          py::arg("boundaries"),
          "Write nodes and element data to a meshb file.");

    /**
     * Read solution data from a GMF solution (.solb) file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @param labelpath Path to the map between solution strings and ref IDs
     * @return Dictionary of solution fields
     */
    m.def("read_solution", &pymeshb::gmf::read_solution,
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
     * @param labelpath Path to the map between solution strings and ref IDs
     * @return Boolean indicating success
     */
    m.def("write_solution", &pymeshb::gmf::write_solution,
          py::arg("solpath"), 
          py::arg("sol_data"), 
          py::arg("num_ver"),
          py::arg("dim"), 
          py::arg("labelpath") = "", 
          "Write solution data to a solb solution file.");
}