#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <numeric>

#include "../include/refine_3d.hpp"

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

// Helper function to check and fix tetrahedron orientation
// VTK convention: (p1-p0) x (p2-p0) dot (p3-p0) < 0
inline void check_tet(unsigned int& v0, unsigned int& v1, unsigned int& v2, unsigned int& v3,
                      const double* coords, int dim) {
    // Get coordinates
    double p0[3] = {coords[v0 * dim], coords[v0 * dim + 1], coords[v0 * dim + 2]};
    double p1[3] = {coords[v1 * dim], coords[v1 * dim + 1], coords[v1 * dim + 2]};
    double p2[3] = {coords[v2 * dim], coords[v2 * dim + 1], coords[v2 * dim + 2]};
    double p3[3] = {coords[v3 * dim], coords[v3 * dim + 1], coords[v3 * dim + 2]};

    // Compute vectors
    double v10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    double v20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    double v30[3] = {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};

    // Cross product: v10 x v20
    double cross[3] = {
        v10[1] * v20[2] - v10[2] * v20[1],
        v10[2] * v20[0] - v10[0] * v20[2],
        v10[0] * v20[1] - v10[1] * v20[0]
    };

    // Dot product: (v10 x v20) · v30
    double det = cross[0] * v30[0] + cross[1] * v30[1] + cross[2] * v30[2];

    // If positive, orientation is wrong - swap v1 and v2
    if (det > 0) {
        std::swap(v1, v2);
    }
}

py::dict refine_3d(const py::dict& mesh_data) {
    // Validate required keys
    if (!mesh_data.contains("coords")) {
        throw std::runtime_error("mesh_data must contain 'coords' key");
    }
    if (!mesh_data.contains("elements")) {
        throw std::runtime_error("mesh_data must contain 'elements' key");
    }
    if (!mesh_data.contains("boundaries")) {
        throw std::runtime_error("mesh_data must contain 'boundaries' key");
    }

    // Extract data from dictionary
    py::array_t<double> coords = mesh_data["coords"].cast<py::array_t<double>>();
    py::dict elements = mesh_data["elements"].cast<py::dict>();
    py::dict boundaries = mesh_data["boundaries"].cast<py::dict>();

    auto coords_buf = coords.request();
    auto* coords_ptr = static_cast<double*>(coords_buf.ptr);
    const int num_pts = coords_buf.shape[0];
    const int dim = coords_buf.shape[1];

    // Assert that the mesh is 3D
    if (dim != 3) {
        throw std::runtime_error("Input mesh must be 3D.");
    }

    // Validate input element and boundary types
    for (auto item : elements) {
        std::string key = item.first.cast<std::string>();
        if (key != "Tetrahedra") {
            throw std::runtime_error("Invalid element type in 3D refinement: " + key + ". Only 'Tetrahedra' is supported.");
        }
    }
    for (auto item : boundaries) {
        std::string key = item.first.cast<std::string>();
        if (key != "Triangles") {
            throw std::runtime_error("Invalid boundary type in 3D refinement: " + key + ". Only 'Triangles' is supported.");
        }
    }

    std::vector<double> new_coords_vec(coords_ptr, coords_ptr + num_pts * dim);
    std::map<Edge, unsigned int> edge_to_midpoint;

    py::dict new_elements;
    py::dict new_boundaries;

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
                int num_edges = (key == "Tetrahedra") ? 6 : 3;
                std::vector<std::pair<int, int>> edge_pairs;
                if (key == "Tetrahedra") {
                    edge_pairs = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
                } else { // Triangles
                    edge_pairs = {{0, 1}, {1, 2}, {2, 0}};
                }

                for (const auto& p : edge_pairs) {
                    const unsigned int p0 = elem_nodes[p.first];
                    const unsigned int p1 = elem_nodes[p.second];
                    Edge edge = {p0, p1};

                    if (edge_to_midpoint.find(edge) == edge_to_midpoint.end()) {
                        const double mx = (coords_ptr[p0 * dim] + coords_ptr[p1 * dim]) / 2.0;
                        const double my = (coords_ptr[p0 * dim + 1] + coords_ptr[p1 * dim + 1]) / 2.0;
                        const double mz = (coords_ptr[p0 * dim + 2] + coords_ptr[p1 * dim + 2]) / 2.0;
                        new_coords_vec.push_back(mx);
                        new_coords_vec.push_back(my);
                        new_coords_vec.push_back(mz);
                        unsigned int midpoint_idx = (new_coords_vec.size() / dim) - 1;
                        edge_to_midpoint[edge] = midpoint_idx;
                        midpoint_indices.push_back(midpoint_idx);
                    } else {
                        midpoint_indices.push_back(edge_to_midpoint[edge]);
                    }
                }

                if (key == "Tetrahedra" && !is_boundary) {
                    const unsigned int p0 = elem_nodes[0], p1 = elem_nodes[1], p2 = elem_nodes[2], p3 = elem_nodes[3];
                    const unsigned int m01 = midpoint_indices[0], m02 = midpoint_indices[1], m03 = midpoint_indices[2];
                    const unsigned int m12 = midpoint_indices[3], m13 = midpoint_indices[4], m23 = midpoint_indices[5];

                    // Centroid
                    double cx = 0, cy = 0, cz = 0;
                    for (int k = 0; k < 4; ++k) {
                        cx += coords_ptr[elem_nodes[k] * dim];
                        cy += coords_ptr[elem_nodes[k] * dim + 1];
                        cz += coords_ptr[elem_nodes[k] * dim + 2];
                    }
                    new_coords_vec.push_back(cx / 4.0);
                    new_coords_vec.push_back(cy / 4.0);
                    new_coords_vec.push_back(cz / 4.0);
                    const unsigned int center_idx = (new_coords_vec.size() / dim) - 1;

                    // Create 8 new tetrahedra with proper orientation
                    // Need to use new_coords_vec for the orientation check since we've added new points
                    const double* all_coords = new_coords_vec.data();

                    // Corner tets (4)
                    std::vector<std::array<unsigned int, 4>> tet_verts = {
                        {p0, m01, m02, m03},
                        {p1, m12, m01, m13},
                        {p2, m23, m12, m02},
                        {p3, m03, m23, m13},
                        // Interior octahedron tets (4)
                        {m01, m12, m02, center_idx},
                        {m02, m12, m23, center_idx},
                        {m03, m23, m02, center_idx},
                        {m01, m03, m13, center_idx}
                    };

                    for (auto& tet : tet_verts) {
                        check_tet(tet[0], tet[1], tet[2], tet[3], all_coords, dim);
                        new_elem_vec.push_back(tet[0]);
                        new_elem_vec.push_back(tet[1]);
                        new_elem_vec.push_back(tet[2]);
                        new_elem_vec.push_back(tet[3]);
                        new_elem_vec.push_back(ref);
                    }

                } else if (key == "Triangles" && is_boundary) {
                    const unsigned int p0 = elem_nodes[0], p1 = elem_nodes[1], p2 = elem_nodes[2];
                    const unsigned int m01 = midpoint_indices[0], m12 = midpoint_indices[1], m20 = midpoint_indices[2];

                    unsigned int new_tris[] = {
                        p0, m01, m20, ref,
                        m01, p1, m12, ref,
                        m20, m12, p2, ref,
                        m01, m12, m20, ref
                    };
                    new_elem_vec.insert(new_elem_vec.end(), new_tris, new_tris + 16);
                }
            }

            if (!new_elem_vec.empty()) {
                int cols = 0;
                if (key == "Tetrahedra") {
                    cols = 5;
                } else if (key == "Triangles") {
                    cols = 4;
                }

                int rows = new_elem_vec.size() / cols;
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

    py::dict new_mesh_data;
    new_mesh_data["coords"] = new_coords;
    new_mesh_data["elements"] = new_elements;
    new_mesh_data["boundaries"] = new_boundaries;
    new_mesh_data["dim"] = dim;
    new_mesh_data["num_point"] = new_num_pts;

    return new_mesh_data;
}

} // namespace mesh
} // namespace limon
