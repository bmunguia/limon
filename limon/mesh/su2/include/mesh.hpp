#ifndef LIMON_SU2_MESH_H
#define LIMON_SU2_MESH_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>

namespace py = pybind11;

namespace limon {
namespace su2 {

/**
 * Read mesh data from an SU2 file.
 *
 * @param meshpath Path to the mesh file
 * @param write_markers Whether to write the markers to the file specified in markerpath
 * @param markerpath Path to write the marker reference map file (optional)
 * @return Tuple of (mesh_data dict, marker_map dict)
 */
py::tuple load_mesh(const std::string& meshpath, bool write_markers = false, const std::string& markerpath = "");

/**
 * Write mesh data to an SU2 mesh file.
 *
 * @param meshpath Path to the mesh file
 * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
 * @param marker_map Dictionary mapping marker IDs to marker names
 * @return Boolean indicating success
 */
bool write_mesh(const std::string& meshpath, const py::dict& mesh_data, const py::dict& marker_map);

}  // namespace su2
}  // namespace limon

#endif  // LIMON_SU2_MESH_H