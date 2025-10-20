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
      * @param markerpath Path to the map between marker strings and ref IDs (optional)
      * @param write_markers Whether to write the markers to the file specified in markerpath
      * @return Dictionary containing mesh data with keys: coords, elements, boundaries, dim, num_point
      */
     m.def("load_mesh", &limon::su2::load_mesh,
          py::arg("meshpath"),
          py::arg("markerpath") = "",
          py::arg("write_markers") = false,
          "Read a SU2 mesh file and return mesh data as a dictionary.");

     /**
      * Write mesh data to a SU2 mesh (.su2) file.
      *
      * @param meshpath Path to the mesh file
      * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
      * @param markerpath Path to the map between marker strings and ref IDs (optional)
      * @return Boolean indicating success
      */
     m.def("write_mesh", &limon::su2::write_mesh,
          py::arg("meshpath"),
          py::arg("mesh_data"),
          py::arg("markerpath") = "",
          "Write mesh data to a SU2 mesh file.");

    /**
     * Read solution data from a SU2 solution (.dat) file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @param labelpath Path to the map between solution strings and ref IDs
   * @param write_labels Whether to write the labels to the file specified in labelpath
     * @return Dictionary of solution fields
     */
    m.def("load_solution", &limon::su2::load_solution,
        py::arg("solpath"),
        py::arg("num_ver"),
        py::arg("dim"),
        py::arg("labelpath") = "",
        py::arg("write_labels") = false,
        "Read a SU2 solution file and return solution fields as a dictionary");

  /**
   * Write solution data to a SU2 solution (.dat) file.
   *
   * @param solpath Path to the solution file
   * @param sol Dictionary of solution fields
   * @param num_ver Number of vertices
   * @param dim Mesh dimension
   * @param labelpath Path to the map between solution strings and ref IDs
   * @return Boolean indicating success
   */
  m.def("write_solution", &limon::su2::write_solution,
        py::arg("solpath"),
        py::arg("sol"),
        py::arg("num_ver"),
        py::arg("dim"),
        py::arg("labelpath") = "",
        "Write solution fields to a SU2 solution file");
 }