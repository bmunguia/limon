#ifndef PYMESHB_GMF_ELEMENT_HPP
#define PYMESHB_GMF_ELEMENT_HPP

#include <cstdint>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace pymeshb {
namespace gmf {

void read_element_type(int64_t mesh_id, int kwd, int num_nodes, py::dict& elements, const std::string& key_name);
void read_elements_2D(int64_t mesh_id, py::dict& elements);
void read_elements_3D(int64_t mesh_id, py::dict& elements);

void write_element_type(int64_t mesh_id, int kwd, py::array_t<unsigned int>& element_array);
void write_elements_2D(int64_t mesh_id, py::dict& elements);
void write_elements_3D(int64_t mesh_id, py::dict& elements);

} // namespace gmf
} // namespace pymeshb

#endif // PYMESHB_GMF_ELEMENT_HPP