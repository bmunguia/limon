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
 * @param markerpath Path to the map between marker strings and ref IDs (optional)
 * @param write_markers Whether to write the markers to the file specified in markerpath
 * @return Dictionary containing mesh data with keys: coords, elements, boundaries, dim, num_point
 */
py::dict load_mesh(const std::string& meshpath, const std::string& markerpath = "", bool write_markers = false);

/**
 * Write mesh data to an SU2 mesh file.
 *
 * @param meshpath Path to the mesh file
 * @param mesh_data Dictionary containing mesh data with keys: coords, elements, boundaries
 * @param markerpath Path to the map between marker strings and ref IDs (optional)
 * @return Boolean indicating success
 */
bool write_mesh(const std::string& meshpath, const py::dict& mesh_data,
                const std::string& markerpath = "");

}  // namespace su2
}  // namespace limon

#endif  // LIMON_SU2_MESH_H