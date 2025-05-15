#ifndef PYMESHB_SU2_MESH_H
#define PYMESHB_SU2_MESH_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>

namespace py = pybind11;

namespace pymeshb {
namespace su2 {

/**
 * Read mesh data from an SU2 file.
 *
 * @param meshpath Path to the mesh file
 * @param solpath Path to the solution file (optional)
 * @param read_sol Whether to read solution data (default: false)
 * @return Tuple containing coordinates, elements, and optionally solution data
 */
py::tuple read_mesh(const std::string& meshpath, const std::string& solpath = "",
                    bool read_sol = false);

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
bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                py::dict elements, const std::string& solpath = "",
                py::dict sol = py::dict());

}  // namespace su2
}  // namespace pymeshb

#endif  // PYMESHB_SU2_MESH_H