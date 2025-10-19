#ifndef LIMON_GMF_ELEMENT_HPP
#define LIMON_GMF_ELEMENT_HPP

#include <cstdint>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace limon {
namespace gmf {

void read_element_type(int64_t mesh_id, int kwd, int num_nodes, py::dict& elements, const std::string& key_name);
void read_elements_2D(int64_t mesh_id, py::dict& elements, py::dict& boundaries);
void read_elements_3D(int64_t mesh_id, py::dict& elements, py::dict& boundaries);

void write_element_type(int64_t mesh_id, int kwd, py::array_t<unsigned int>& element_array);
void write_elements_2D(int64_t mesh_id, const py::dict& elements, const py::dict& boundaries);
void write_elements_3D(int64_t mesh_id, const py::dict& elements, const py::dict& boundaries);

} // namespace gmf
} // namespace limon

#endif // LIMON_GMF_ELEMENT_HPP