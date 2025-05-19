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

 #include "mesh.hpp"
 #include "solution.hpp"

 namespace py = pybind11;

 PYBIND11_MODULE(libsu2, m) {
     m.doc() = "Python bindings for SU2 mesh and solution files";

     /**
      * Read mesh data from a SU2 mesh (.su2) file.
      *
      * @param meshpath Path to the mesh file
      * @param solpath Path to the solution file (optional)
      * @param read_sol Whether to read solution data (default: false)
      * @return Tuple containing coordinates, elements, and optionally solution data
      */
     m.def("read_mesh", &pymeshb::su2::read_mesh,
          py::arg("meshpath"), py::arg("solpath") = "", py::arg("read_sol") = false,
          "Read a SU2 mesh file and return nodes, elements, and optionally solution data.");

     /**
      * Write mesh data to a SU2 mesh (.su2) file.
      *
      * @param meshpath Path to the mesh file
      * @param coords Coordinates of each node
      * @param elements Dictionary of mesh elements
      * @param solpath Path to the solution file (optional)
      * @param sol Dictionary of solution data (optional)
      * @return Boolean indicating success
      */
     m.def("write_mesh", &pymeshb::su2::write_mesh,
          py::arg("meshpath"), py::arg("coords"), py::arg("elements"),
          py::arg("boundaries"), py::arg("solpath") = "", py::arg("sol") = py::dict(),
          "Write nodes, elements, and optionally solution data to a SU2 mesh file.");

    /**
     * Read solution data from a SU2 solution (.dat) file.
     *
     * @param solpath Path to the solution file
     * @param num_ver Number of vertices
     * @param dim Mesh dimension
     * @return Dictionary of solution fields
     */
    m.def("read_solution", &pymeshb::su2::read_solution,
        py::arg("solpath"), py::arg("num_ver"), py::arg("dim"),
        "Read a SU2 solution file and return solution fields as a dictionary");

  /**
   * Write solution data to a SU2 solution (.dat) file.
   *
   * @param solpath Path to the solution file
   * @param sol Dictionary of solution fields
   * @param num_ver Number of vertices
   * @param dim Mesh dimension
   * @return Boolean indicating success
   */
  m.def("write_solution", &pymeshb::su2::write_solution,
        py::arg("solpath"), py::arg("sol"), py::arg("num_ver"), py::arg("dim"),
        "Write solution fields to a SU2 solution file");
 }