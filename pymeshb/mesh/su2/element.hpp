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

bool read_elements(
    std::ifstream& file_stream,
    int elem_count,
    py::dict& elements
);

bool read_boundary_elements(
    std::ifstream& file_stream,
    int boundary_count,
    py::dict& boundaries
);

int process_elements_for_writing(
    const py::dict& elements,
    const py::dict& boundaries,
    int dim,
    std::map<std::string, std::vector<std::vector<unsigned int>>>& interior_elements,
    std::map<int, std::vector<std::vector<unsigned int>>>& boundary_elements
);

int write_elements(
    std::ofstream& mesh_file,
    const std::map<std::string, std::vector<std::vector<unsigned int>>>& interior_elements,
    const std::map<std::string, int>& elem_type_map
);

bool write_boundary_elements(
    std::ofstream& mesh_file,
    const std::map<int, std::vector<std::vector<unsigned int>>>& boundary_elements
);

} // namespace su2
} // namespace pymeshb

#endif // PYMESHB_SU2_ELEMENT_HPP