#ifndef LIMON_GMF_SOLUTION_HPP
#define LIMON_GMF_SOLUTION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace limon {
namespace gmf {

/**
 * Read solution data from a GMF solution (.solb) file.
 *
 * @param solpath Path to the solution file
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @param label_map Dictionary mapping label IDs to label names (optional)
 * @param read_labels Whether to read labels from labelpath file
 * @param labelpath Path to the label reference map file
 * @return Tuple of (solution dict, label_map dict)
 */
py::tuple load_solution(const std::string& solpath, int64_t num_ver, int dim,
                        py::dict label_map, bool read_labels, const std::string& labelpath);

/**
 * Write solution data to a GMF solution (.solb) file.
 *
 * @param solpath Path to the solution file
 * @param sol Dictionary of solution fields
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @return Boolean indicating success
 */
bool write_solution(const std::string& solpath, py::dict sol_data, int64_t num_ver, int dim);

} // namespace gmf
} // namespace limon

#endif // LIMON_GMF_SOLUTION_HPP