#ifndef PYMESHB_SU2_ELEMENT_HPP
#define PYMESHB_SU2_ELEMENT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace py = pybind11;

namespace pymeshb {
namespace su2 {

struct ElementTypeInfo {
    std::string name;     // Element name
    int num_node;         // Number of nodes per element
};

std::map<int, ElementTypeInfo> get_element_type_map();

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

} // namespace su2
} // namespace pymeshb

#endif // PYMESHB_SU2_ELEMENT_HPP