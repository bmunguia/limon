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
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @param labelpath Path to the map between solution strings and ref IDs
 * @return Dictionary of solution fields
 */
py::dict read_solution(const std::string& solpath, int64_t num_ver, int dim,
                       const std::string& labelpath);

/**
 * Write solution data to a GMF solution (.solb) file.
 *
 * @param solpath Path to the solution file
 * @param sol Dictionary of solution fields
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @param labelpath Path to the map between solution strings and ref IDs
 * @return Boolean indicating success
 */
bool write_solution(const std::string& solpath, py::dict sol_data, int64_t num_ver, int dim, 
                    const std::string& labelpath);

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_SOLUTION_HPP