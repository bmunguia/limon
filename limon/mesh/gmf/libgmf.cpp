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
     * @param marker_map Dictionary mapping marker IDs to names (optional)
     * @param read_markers Whether to read markers from markerpath file
     * @param markerpath Path to the marker reference map file
     * @return Tuple of (mesh_data dict, marker_map dict)
     */
    m.def("load_mesh", &limon::gmf::load_mesh,
          py::arg("meshpath"),
          py::arg("marker_map") = py::dict(),
          py::arg("read_markers") = false,
          py::arg("markerpath") = "",
          "Read a meshb file and return tuple of (mesh_data, marker_map).");

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
     * @param label_map Dictionary mapping label IDs to names (optional)
     * @param read_labels Whether to read labels from labelpath file
     * @param labelpath Path to the label reference map file
     * @return Tuple of (solution dict, label_map dict)
     */
    m.def("load_solution", &limon::gmf::load_solution,
          py::arg("solpath"),
          py::arg("num_ver"),
          py::arg("dim"),
          py::arg("label_map") = py::dict(),
          py::arg("read_labels") = false,
          py::arg("labelpath") = "",
          "Read a solb solution file and return tuple of (solution, label_map).");

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