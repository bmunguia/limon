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
 * @param marker_map Dictionary mapping marker IDs to marker names (optional)
 * @param read_markers Whether to read markers from markerpath file
 * @param markerpath Path to the marker reference map file
 * @return Tuple of (mesh_data dict, marker_map dict)
 */
py::tuple load_mesh(const std::string& meshpath, py::dict marker_map,
                    bool read_markers, const std::string& markerpath);

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