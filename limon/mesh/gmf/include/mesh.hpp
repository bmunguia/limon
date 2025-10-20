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
 * @return Dictionary containing mesh data with keys: coords, elements, boundaries, dim, num_point
 */
py::dict load_mesh(const std::string& meshpath);

/**
 * Write mesh data to a GMF mesh (.meshb) file.
 *
 * @param meshpath Path to the mesh file
 * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
 * @return Boolean indicating success
 */
bool write_mesh(const std::string& meshpath, const py::dict& mesh_data);

} // namespace gmf
} // namespace limon

#endif // LIMON_GMF_MESH_HPP