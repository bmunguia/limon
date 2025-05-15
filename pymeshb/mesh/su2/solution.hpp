#ifndef PYMESHB_SU2_SOLUTION_H
#define PYMESHB_SU2_SOLUTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pymeshb {
namespace su2 {

/**
 * Read solution data from an SU2 solution file.
 *
 * @param solpath Path to the solution file
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @return Dictionary of solution fields
 */
py::dict read_solution(const std::string& solpath, int num_ver, int dim);

/**
 * Write solution data to an SU2 solution file.
 *
 * @param solpath Path to the solution file
 * @param sol Dictionary of solution fields
 * @param num_ver Number of vertices
 * @param dim Mesh dimension
 * @return Boolean indicating success
 */
bool write_solution(const std::string& solpath, py::dict sol, int num_ver, int dim);

}  // namespace su2
}  // namespace pymeshb

#endif  // PYMESHB_SU2_SOLUTION_H