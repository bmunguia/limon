#include "include/refine_2d.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace limon {
namespace mesh {

// Helper struct for edges, used to find midpoints
struct Edge {
    unsigned int p0, p1;

    bool operator==(const Edge& other) const {
        return (p0 == other.p0 && p1 == other.p1) || (p0 == other.p1 && p1 == other.p0);
    }

    bool operator<(const Edge& other) const {
        return std::tie(std::min(p0, p1), std::max(p0, p1)) <
               std::tie(std::min(other.p0, other.p1), std::max(other.p0, other.p1));
    }
};


py::tuple refine_2d(py::array_t<double> coords, const py::dict& elements, const py::dict& boundaries) {

    auto coords_buf = coords.request();
    auto* coords_ptr = static_cast<double*>(coords_buf.ptr);
    const int num_pts = coords_buf.shape[0];
    const int dim = coords_buf.shape[1];

    // Assert that the mesh is 2D
    if (dim != 2) {
        throw std::runtime_error("Input mesh must be 2D.");
    }

    // Validate input element and boundary types
    for (auto item : elements) {
        std::string key = item.first.cast<std::string>();
        if (key != "Triangles" && key != "Quadrilaterals") {
            throw std::runtime_error("Invalid element type in 2D refinement: " + key);
        }
    }
    for (auto item : boundaries) {
        std::string key = item.first.cast<std::string>();
        if (key != "Edges") {
            throw std::runtime_error("Invalid boundary type in 2D refinement: " + key);
        }
    }

    // Build connectivity and storage for new mesh data
    // - `new_coords_vec` starts with original points and grows as new midpoints are added
    // - `edge_to_midpoint` maps an edge to the index of its new midpoint to avoid creating duplicate points
    std::vector<double> new_coords_vec(coords_ptr, coords_ptr + num_pts * dim);
    std::map<Edge, unsigned int> edge_to_midpoint;

    py::dict new_elements;
    py::dict new_boundaries;

    // Process both volume elements and boundary elements with the same logic
    auto process_elements = [&](const py::dict& element_dict, bool is_boundary) {
        for (auto item : element_dict) {
            std::string key = item.first.cast<std::string>();
            py::array_t<unsigned int> element_array = item.second.cast<py::array_t<unsigned int>>();
            auto elem_buf = element_array.request();
            auto* elem_ptr = static_cast<unsigned int*>(elem_buf.ptr);
            int num_elem = elem_buf.shape[0];
            int num_node = elem_buf.shape[1] - 1;

            std::vector<unsigned int> new_elem_vec;

            for (int i = 0; i < num_elem; ++i) {
                std::vector<unsigned int> elem_nodes(num_node);
                unsigned int ref = elem_ptr[i * (num_node + 1) + num_node];
                for (int j = 0; j < num_node; ++j) {
                    elem_nodes[j] = elem_ptr[i * (num_node + 1) + j];
                }

                std::vector<unsigned int> midpoint_indices;
                for (int j = 0; j < num_node; ++j) {
                    const unsigned int p0 = elem_nodes[j];
                    const unsigned int p1 = elem_nodes[(j + 1) % num_node];
                    Edge edge = {p0, p1};

                    if (edge_to_midpoint.find(edge) == edge_to_midpoint.end()) {
                        const double mx = (coords_ptr[p0 * dim] + coords_ptr[p1 * dim]) / 2.0;
                        const double my = (coords_ptr[p0 * dim + 1] + coords_ptr[p1 * dim + 1]) / 2.0;
                        new_coords_vec.push_back(mx);
                        new_coords_vec.push_back(my);
                        unsigned int midpoint_idx = (new_coords_vec.size() / dim) - 1;
                        edge_to_midpoint[edge] = midpoint_idx;
                        midpoint_indices.push_back(midpoint_idx);
                    } else {
                        midpoint_indices.push_back(edge_to_midpoint[edge]);
                    }
                }

                if (key == "Triangles" && !is_boundary) {
                    // 4 new triangles using original vertices and edge midpoints
                    const unsigned int p0 = elem_nodes[0];
                    const unsigned int p1 = elem_nodes[1];
                    const unsigned int p2 = elem_nodes[2];
                    const unsigned int m01 = midpoint_indices[0];
                    const unsigned int m12 = midpoint_indices[1];
                    const unsigned int m20 = midpoint_indices[2];

                    unsigned int new_tris[] = {
                        p0, m01, m20, ref,
                        m01, p1, m12, ref,
                        m20, m12, p2, ref,
                        m01, m12, m20, ref
                    };
                    new_elem_vec.insert(new_elem_vec.end(), new_tris, new_tris + 16);
                } else if (key == "Quadrilaterals" && !is_boundary) {
                    // 4 new quads using original vertices, edge midpoints, and centroid
                    const unsigned int p0 = elem_nodes[0];
                    const unsigned int p1 = elem_nodes[1];
                    const unsigned int p2 = elem_nodes[2];
                    const unsigned int p3 = elem_nodes[3];
                    const unsigned int m01 = midpoint_indices[0];
                    const unsigned int m12 = midpoint_indices[1];
                    const unsigned int m23 = midpoint_indices[2];
                    const unsigned int m30 = midpoint_indices[3];

                    // Centroid
                    double cx = 0, cy = 0;
                    for (int k = 0; k < 4; ++k) {
                        cx += coords_ptr[elem_nodes[k] * dim];
                        cy += coords_ptr[elem_nodes[k] * dim + 1];
                    }
                    new_coords_vec.push_back(cx / 4.0);
                    new_coords_vec.push_back(cy / 4.0);
                    const unsigned int center_idx = (new_coords_vec.size() / dim) - 1;

                    unsigned int new_quads[] = {
                        p0, m01, center_idx, m30, ref,
                        m01, p1, m12, center_idx, ref,
                        center_idx, m12, p2, m23, ref,
                        m30, center_idx, m23, p3, ref
                    };
                    new_elem_vec.insert(new_elem_vec.end(), new_quads, new_quads + 20);
                } else if (key == "Edges" && is_boundary) {
                    // 2 new edges using endpoints and midpoint
                    const unsigned int p0 = elem_nodes[0], p1 = elem_nodes[1];
                    const unsigned int m = midpoint_indices[0];
                    unsigned int new_edges_arr[] = {
                        p0, m, ref,
                        m, p1, ref
                    };
                    new_elem_vec.insert(new_elem_vec.end(), new_edges_arr, new_edges_arr + 6);
                }
            }

            if (!new_elem_vec.empty()) {
                // Number of nodes per element + 1 for the reference ID
                int cols = 0;
                if (key == "Triangles") {
                    cols = 4;
                } else if (key == "Quadrilaterals") {
                    cols = 5;
                } else if (key == "Edges") {
                    cols = 3;
                }

                // Number of new elements
                int rows = new_elem_vec.size() / cols;

                // Create the new 2D NumPy array
                py::array_t<unsigned int> new_array({rows, cols}, new_elem_vec.data());

                if (is_boundary) {
                    new_boundaries[key.c_str()] = new_array;
                } else {
                    new_elements[key.c_str()] = new_array;
                }
            }
        }
    };

    process_elements(elements, false);
    process_elements(boundaries, true);

    int new_num_pts = new_coords_vec.size() / dim;
    py::array_t<double> new_coords({new_num_pts, dim}, new_coords_vec.data());

    return py::make_tuple(new_coords, new_elements, new_boundaries);
}

} // namespace mesh
} // namespace limon
