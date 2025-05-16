#ifndef PYMESHB_GMF_MESH_HPP
#define PYMESHB_GMF_MESH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pymeshb {
namespace gmf {

/**
 * Read mesh data from a GMF mesh (.meshb) file.
 *
 * @param meshpath Path to the mesh file
 * @param solpath Path to the solution file (optional)
 * @param read_sol Whether to read solution data (default: false)
 * @return Tuple containing coordinates, elements, and optionally solution data
 */
py::tuple read_mesh(const std::string& meshpath, const std::string& solpath = "",
                    bool read_sol = false);

/**
 * Write mesh data to a GMF mesh (.meshb) file.
 *
 * @param meshpath Path to the mesh file
 * @param coords Coordinates of each node
 * @param elements Dictionary of mesh elements
 * @param boundaries Dictionary of mesh boundary elements
 * @param solpath Path to the solution file (optional)
 * @param sol Dictionary of solution data (optional)
 * @return Boolean indicating success
 */
bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                const py::dict& elements, const py::dict& boundaries,
                const std::string& solpath = "", py::dict sol = py::dict());

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_MESH_HPP