#ifndef LIMON_GMF_MESH_HPP
#define LIMON_GMF_MESH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace limon {
namespace gmf {

/**
 * Read mesh data from a GMF mesh (.meshb) file.
 *
 * @param meshpath Path to the mesh file
 * @return Tuple containing coordinates, elements, and optionally solution data
 */
py::tuple load_mesh(const std::string& meshpath);

/**
 * Write mesh data to a GMF mesh (.meshb) file.
 *
 * @param meshpath Path to the mesh file
 * @param coords Coordinates of each node
 * @param elements Dictionary of mesh elements
 * @param boundaries Dictionary of mesh boundary elements
 * @return Boolean indicating success
 */
bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                const py::dict& elements, const py::dict& boundaries);

} // namespace gmf
} // namespace limon

#endif // LIMON_GMF_MESH_HPP