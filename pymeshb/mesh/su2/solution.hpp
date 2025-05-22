#ifndef PYMESHB_SU2_SOLUTION_H
#define PYMESHB_SU2_SOLUTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pymeshb {
namespace su2 {

/**
 * Read solution data from an SU2 solution file.
 *
 * @param solpath Path to the solution file
 * @param labelpath Path to the map between solution strings and ref IDs
 * @param num_point Number of vertices
 * @param dim Mesh dimension
 * @return Dictionary of solution fields
 */
py::dict read_solution(const std::string& solpath, const std::string& labelpath, int num_point, int dim);

py::dict read_solution_ascii(const std::string& solpath, int num_point, int dim);

py::dict read_solution_binary(const std::string& solpath, int num_point, int dim);

py::dict process_solution_fields(const std::vector<std::string>& field_names,
                                 const std::vector<std::vector<double>>& data,
                                 int num_point, int dim);

/**
 * Write solution data to an SU2 solution file.
 *
 * @param solpath Path to the solution file
 * @param labelpath Path to the map between solution strings and ref IDs
 * @param sol Dictionary of solution fields
 * @param num_point Number of vertices
 * @param dim Mesh dimension
 * @return Boolean indicating success
 */
bool write_solution(const std::string& solpath, const std::string& labelpath, py::dict sol, int num_point, int dim);

bool write_solution_ascii(const std::string& solpath, py::dict sol, int num_ver, int dim);

bool write_solution_binary(const std::string& solpath, py::dict sol, int num_point, int dim);

}  // namespace su2
}  // namespace pymeshb

#endif  // PYMESHB_SU2_SOLUTION_H