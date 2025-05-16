#ifndef PYMESHB_GMF_MESH_HPP
#define PYMESHB_GMF_MESH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pymeshb {
namespace gmf {

py::tuple read_mesh(const std::string& meshpath, const std::string& solpath = "", 
                    bool read_sol = false);

bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                    py::dict elements, const std::string& solpath = "",
                    py::dict sol = py::dict());

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_MESH_HPP