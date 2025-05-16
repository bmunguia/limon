#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}
#include "element.hpp"

namespace pymeshb {
namespace gmf {

void read_element_type(int64_t mesh_id, int kwd, int num_node, py::dict& elements, const std::string& key_name) {
    int64_t num_elm = GmfStatKwd(mesh_id, kwd);
    if (num_elm > 0) {
        py::array_t<unsigned int> element_array(std::vector<py::ssize_t>{num_elm,
                                                static_cast<py::ssize_t>(num_node) + 1});
        auto elm_ptr = element_array.mutable_data();

        if (GmfGotoKwd(mesh_id, kwd)) {
            for (auto i = 0; i < num_elm; i++) {
                std::vector<int> bufInt(num_node);
                int ref;
                switch (kwd) {
                    case GmfEdges:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &ref);
                        break;
                    case GmfTriangles:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &ref);
                        break;
                    case GmfQuadrilaterals:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2],
                                  &bufInt[3], &ref);
                        break;
                    case GmfTetrahedra:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2],
                                  &bufInt[3], &ref);
                        break;
                    case GmfPrisms:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2],
                                  &bufInt[3], &bufInt[4], &bufInt[5], &ref);
                        break;
                    case GmfHexahedra:
                        GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2],
                                  &bufInt[3], &bufInt[4], &bufInt[5], &bufInt[6], &bufInt[7], &ref);
                        break;
                    default:
                        throw std::runtime_error("Unsupported keyword in read_element_type for GmfGetLin: " + std::to_string(kwd));
                }

                for (auto j = 0; j < num_node; j++) {
                    elm_ptr[i * (num_node + 1) + j] = bufInt[j];
                }
                elm_ptr[i * (num_node + 1) + num_node] = ref;
            }
        }
        elements[key_name.c_str()] = element_array;
    }
}

void read_elements_2D(int64_t mesh_id, py::dict& elements) {
    read_element_type(mesh_id, GmfEdges, 2, elements, "Edges");
    read_element_type(mesh_id, GmfTriangles, 3, elements, "Triangles");
    read_element_type(mesh_id, GmfQuadrilaterals, 4, elements, "Quadrilaterals");
}

void read_elements_3D(int64_t mesh_id, py::dict& elements) {
    read_element_type(mesh_id, GmfTriangles, 3, elements, "Triangles");
    read_element_type(mesh_id, GmfQuadrilaterals, 4, elements, "Quadrilaterals");
    read_element_type(mesh_id, GmfTetrahedra, 4, elements, "Tetrahedra");
    read_element_type(mesh_id, GmfPrisms, 6, elements, "Prisms");
    read_element_type(mesh_id, GmfHexahedra, 8, elements, "Hexahedra");
}

void write_element_type(int64_t mesh_id, int kwd, py::array_t<unsigned int>& element_array) {
    auto elm_buf = element_array.request();
    unsigned int* elm_ptr = static_cast<unsigned int*>(elm_buf.ptr);
    int64_t num_elm = element_array.shape(0);
    int num_node = element_array.shape(1) - 1;

    GmfSetKwd(mesh_id, kwd, num_elm);
    for (auto i = 0; i < num_elm; i++) {
        int base_idx = i * (num_node + 1);
        switch (kwd) {
            case GmfEdges:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]));
                break;
            case GmfTriangles:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]),
                          static_cast<int>(elm_ptr[base_idx + 3]));
                break;
            case GmfQuadrilaterals:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]),
                          static_cast<int>(elm_ptr[base_idx + 3]),
                          static_cast<int>(elm_ptr[base_idx + 4]));
                break;
            case GmfTetrahedra:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]),
                          static_cast<int>(elm_ptr[base_idx + 3]),
                          static_cast<int>(elm_ptr[base_idx + 4]));
                break;
            case GmfPrisms:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]),
                          static_cast<int>(elm_ptr[base_idx + 3]),
                          static_cast<int>(elm_ptr[base_idx + 4]),
                          static_cast<int>(elm_ptr[base_idx + 5]),
                          static_cast<int>(elm_ptr[base_idx + 6]));
                break;
            case GmfHexahedra:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx]),
                          static_cast<int>(elm_ptr[base_idx + 1]),
                          static_cast<int>(elm_ptr[base_idx + 2]),
                          static_cast<int>(elm_ptr[base_idx + 3]),
                          static_cast<int>(elm_ptr[base_idx + 4]),
                          static_cast<int>(elm_ptr[base_idx + 5]),
                          static_cast<int>(elm_ptr[base_idx + 6]),
                          static_cast<int>(elm_ptr[base_idx + 7]),
                          static_cast<int>(elm_ptr[base_idx + 8]));
                break;
            default:
                throw std::runtime_error("Unsupported keyword in write_element_type for GmfSetLin: " + std::to_string(kwd));
        }
    }
}

void write_elements_2D(int64_t mesh_id, py::dict& elements) {
    if (elements.contains("Edges")) {
        py::array_t<unsigned int> element_array = elements["Edges"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfEdges, element_array);
    }
    if (elements.contains("Triangles")) {
        py::array_t<unsigned int> element_array = elements["Triangles"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfTriangles, element_array);
    }
    if (elements.contains("Quadrilaterals")) {
        py::array_t<unsigned int> element_array = elements["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfQuadrilaterals, element_array);
    }
}

void write_elements_3D(int64_t mesh_id, py::dict& elements) {
    if (elements.contains("Triangles")) {
        py::array_t<unsigned int> element_array = elements["Triangles"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfTriangles, element_array);
    }
    if (elements.contains("Quadrilaterals")) {
        py::array_t<unsigned int> element_array = elements["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfQuadrilaterals, element_array);
    }
    if (elements.contains("Tetrahedra")) {
        py::array_t<unsigned int> element_array = elements["Tetrahedra"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfTetrahedra, element_array);
    }
    if (elements.contains("Prisms")) {
        py::array_t<unsigned int> element_array = elements["Prisms"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfPrisms, element_array);
    }
    if (elements.contains("Hexahedra")) {
        py::array_t<unsigned int> element_array = elements["Hexahedra"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfHexahedra, element_array);
    }
}

} // namespace gmf
} // namespace pymeshb