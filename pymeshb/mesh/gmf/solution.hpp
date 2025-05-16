#ifndef PYMESHB_GMF_SOLUTION_HPP
#define PYMESHB_GMF_SOLUTION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pymeshb {
namespace gmf {

py::dict read_solution(const std::string& solpath, int64_t num_ver, int dim);

bool write_solution(const std::string& solpath, py::dict sol_data, int64_t num_ver, 
                    int dim, int file_version);

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_SOLUTION_HPP