#ifndef PYMESHB_SU2_ELEMENT_HPP
#define PYMESHB_SU2_ELEMENT_HPP

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pymeshb {
namespace su2 {

struct ElementTypeInfo {
    std::string name;     // Element name
    int num_node;         // Number of nodes per element
};

std::map<int, ElementTypeInfo> get_element_type_map();

std::map<std::string, int> get_reverse_element_type_map();

void read_element_type(std::ifstream& file_stream, int elem_type, int count, py::dict& elements,
                       const std::string& key_name);

void read_elements_2D(std::ifstream& file_stream, int elem_count, py::dict& elements);

void read_elements_3D(std::ifstream& file_stream, int elem_count, py::dict& elements);

void read_boundary_element_type(std::ifstream& file_stream, int marker_idx, int marker_elements,
                               int elem_type, py::dict& boundaries, const std::string& key_name);

void read_boundary_elements_2D(std::ifstream& file_stream, int marker_idx, int marker_elements,
                              py::dict& boundaries);

void read_boundary_elements_3D(std::ifstream& file_stream, int marker_idx, int marker_elements,
                              py::dict& boundaries);

bool read_boundary_elements(std::ifstream& file_stream, int boundary_count, py::dict& boundaries,
                            const std::string& markerpath = "");

void write_element_type(std::ofstream& mesh_file, int elem_type, py::array_t<unsigned int>& element_array);

void write_elements_2D(std::ofstream& mesh_file, const py::dict& elements, int& elem_count);

void write_elements_3D(std::ofstream& mesh_file, const py::dict& elements, int& elem_count);

int write_elements(std::ofstream& mesh_file, const py::dict& elements);

void write_boundary_elements_2D(std::ofstream& mesh_file, const py::dict& boundaries,
                                std::map<int, std::string>& ref_map);

void write_boundary_elements_3D(std::ofstream& mesh_file, const py::dict& boundaries,
                                std::map<int, std::string>& ref_map);

int write_boundary_elements(std::ofstream& mesh_file, const py::dict& boundaries,
                            const std::string& markerpath = "");

} // namespace su2
} // namespace pymeshb

#endif // PYMESHB_SU2_ELEMENT_HPP