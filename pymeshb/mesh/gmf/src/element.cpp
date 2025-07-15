#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}
#include "../include/element.hpp"

namespace pymeshb {
namespace gmf {

void read_element_type(int64_t mesh_id, int kwd, int num_node, py::dict& elements, const std::string& key_name) {
    int64_t num_elm = GmfStatKwd(mesh_id, kwd);
    if (num_elm > 0) {
        py::array_t<unsigned int> element_array(std::vector<py::ssize_t>{num_elm,
                                                static_cast<py::ssize_t>(num_node) + 1});
        auto elm_ptr = element_array.mutable_data();
        std::vector<int> bufInt(num_node);

        if (GmfGotoKwd(mesh_id, kwd)) {
            for (auto i = 0; i < num_elm; i++) {
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
                    // GMF uses 1-indexing, so subtract 1 from all indices
                    elm_ptr[i * (num_node + 1) + j] = bufInt[j] - 1;
                }
                elm_ptr[i * (num_node + 1) + num_node] = ref;
            }
        }
        elements[key_name.c_str()] = element_array;
    }
}

void read_elements_2D(int64_t mesh_id, py::dict& elements, py::dict& boundaries) {
    // Boundary elements
    read_element_type(mesh_id, GmfEdges, 2, boundaries, "Edges");

    // Volume elements
    read_element_type(mesh_id, GmfTriangles, 3, elements, "Triangles");
    read_element_type(mesh_id, GmfQuadrilaterals, 4, elements, "Quadrilaterals");
}

void read_elements_3D(int64_t mesh_id, py::dict& elements, py::dict& boundaries) {
    // Boundary elements
    read_element_type(mesh_id, GmfTriangles, 3, boundaries, "Triangles");
    read_element_type(mesh_id, GmfQuadrilaterals, 4, boundaries, "Quadrilaterals");

    // Domain elements
    read_element_type(mesh_id, GmfTetrahedra, 4, elements, "Tetrahedra");
    read_element_type(mesh_id, GmfPrisms, 6, elements, "Prisms");
    read_element_type(mesh_id, GmfHexahedra, 8, elements, "Hexahedra");
}

void write_element_type(int64_t mesh_id, int kwd, py::array_t<unsigned int>& elm_array) {
    auto elm_buf = elm_array.request();
    unsigned int* elm_ptr = static_cast<unsigned int*>(elm_buf.ptr);
    int64_t num_elm = elm_array.shape(0);
    int num_node = elm_array.shape(1) - 1;

    GmfSetKwd(mesh_id, kwd, num_elm);
    for (auto i = 0; i < num_elm; i++) {
        int base_idx = i * (num_node + 1);
        switch (kwd) {
            // GMF uses 1-indexing, so add 1 to all indices
            case GmfEdges:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2]));
                break;
            case GmfTriangles:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2] + 1),
                          static_cast<int>(elm_ptr[base_idx + 3]));
                break;
            case GmfQuadrilaterals:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2] + 1),
                          static_cast<int>(elm_ptr[base_idx + 3] + 1),
                          static_cast<int>(elm_ptr[base_idx + 4]));
                break;
            case GmfTetrahedra:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2] + 1),
                          static_cast<int>(elm_ptr[base_idx + 3] + 1),
                          static_cast<int>(elm_ptr[base_idx + 4]));
                break;
            case GmfPrisms:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2] + 1),
                          static_cast<int>(elm_ptr[base_idx + 3] + 1),
                          static_cast<int>(elm_ptr[base_idx + 4] + 1),
                          static_cast<int>(elm_ptr[base_idx + 5] + 1),
                          static_cast<int>(elm_ptr[base_idx + 6]));
                break;
            case GmfHexahedra:
                GmfSetLin(mesh_id, kwd,
                          static_cast<int>(elm_ptr[base_idx + 0] + 1),
                          static_cast<int>(elm_ptr[base_idx + 1] + 1),
                          static_cast<int>(elm_ptr[base_idx + 2] + 1),
                          static_cast<int>(elm_ptr[base_idx + 3] + 1),
                          static_cast<int>(elm_ptr[base_idx + 4] + 1),
                          static_cast<int>(elm_ptr[base_idx + 5] + 1),
                          static_cast<int>(elm_ptr[base_idx + 6] + 1),
                          static_cast<int>(elm_ptr[base_idx + 7] + 1),
                          static_cast<int>(elm_ptr[base_idx + 8]));
                break;
            default:
                throw std::runtime_error("Unsupported keyword in write_element_type for GmfSetLin: " + std::to_string(kwd));
        }
    }
}

void write_elements_2D(int64_t mesh_id, const py::dict& elements, const py::dict& boundaries) {
    // Boundary elements
    if (boundaries.contains("Edges")) {
        py::array_t<unsigned int> element_array = boundaries["Edges"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfEdges, element_array);
    }

    // Volume elements
    if (elements.contains("Triangles")) {
        py::array_t<unsigned int> element_array = elements["Triangles"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfTriangles, element_array);
    }
    if (elements.contains("Quadrilaterals")) {
        py::array_t<unsigned int> element_array = elements["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfQuadrilaterals, element_array);
    }
}

void write_elements_3D(int64_t mesh_id, const py::dict& elements, const py::dict& boundaries) {
    // Boundary elements
    if (boundaries.contains("Triangles")) {
        py::array_t<unsigned int> element_array = boundaries["Triangles"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfTriangles, element_array);
    }
    if (boundaries.contains("Quadrilaterals")) {
        py::array_t<unsigned int> element_array = boundaries["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_id, GmfQuadrilaterals, element_array);
    }

    // Volume elements
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