/** @file libsu2.cpp
 *  @brief Python bindings for SU2 mesh files.
 *
 *  Implementation of Python bindings for reading and writing SU2 mesh files,
 *  including scalar, vector, and tensor fields in solutions.
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

 PYBIND11_MODULE(libsu2, m) {
     m.doc() = "Python bindings for SU2 mesh and solution files";

     /**
      * Read mesh data from a SU2 mesh (.su2) file.
      *
      * @param meshpath Path to the mesh file
      * @param write_markers Whether to write the markers to the file specified in markerpath
      * @param markerpath Path to write the marker reference map file
      * @return Tuple of (mesh_data dict, marker_map dict)
      */
     m.def("load_mesh", &limon::su2::load_mesh,
          py::arg("meshpath"),
          py::arg("write_markers") = false,
          py::arg("markerpath") = "",
          "Read a SU2 mesh file and return tuple of (mesh_data, marker_map).");

     /**
      * Write mesh data to a SU2 mesh (.su2) file.
      *
      * @param meshpath Path to the mesh file
      * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
      * @param marker_map Dictionary mapping marker IDs to marker names
      * @return Boolean indicating success
      */
     m.def("write_mesh", &limon::su2::write_mesh,
          py::arg("meshpath"),
          py::arg("mesh_data"),
          py::arg("marker_map") = py::dict(),
          "Write mesh data to a SU2 mesh file.");

    /**
     * Read solution data from a SU2 solution (.dat) file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @param write_labels Whether to write the labels to the file specified in labelpath
     * @param labelpath Path to write the label reference map file
     * @return Tuple of (solution dict, label_map dict)
     */
    m.def("load_solution", &limon::su2::load_solution,
        py::arg("solpath"),
        py::arg("num_ver"),
        py::arg("dim"),
        py::arg("write_labels") = false,
        py::arg("labelpath") = "",
        "Read a SU2 solution file and return tuple of (solution, label_map)");

  /**
   * Write solution data to a SU2 solution (.dat) file.
   *
   * @param solpath Path to the solution file
   * @param sol Dictionary of solution fields
   * @param num_ver Number of vertices
   * @param dim Mesh dimension
   * @return Boolean indicating success
   */
  m.def("write_solution", &limon::su2::write_solution,
        py::arg("solpath"),
        py::arg("sol"),
        py::arg("num_ver"),
        py::arg("dim"),
        "Write solution fields to a SU2 solution file");
 }