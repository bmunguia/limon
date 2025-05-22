#ifndef PYMESHB_GMF_SOLUTION_HPP
#define PYMESHB_GMF_SOLUTION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pymeshb {
namespace gmf {

/**
 * Read solution data from a GMF solution (.solb) file.
 *
 * @param solpath Path to the solution file
 * @param labelpath Path to the map between solution strings and ref IDs
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @return Dictionary of solution fields
 */
py::dict read_solution(const std::string& solpath, const std::string& labelpath,
                       int64_t num_ver, int dim);

/**
 * Write solution data to a GMF solution (.solb) file.
 *
 * @param solpath Path to the solution file
 * @param labelpath Path to the map between solution strings and ref IDs
 * @param sol Dictionary of solution fields
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @return Boolean indicating success
 */
bool write_solution(const std::string& solpath, const std::string& labelpath, py::dict sol_data,
                    int64_t num_ver, int dim, int file_version);

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_SOLUTION_HPP