#include <algorithm>
#include <sstream>
#include <stdexcept>
#include "element.hpp"
#include "../util.hpp"

namespace pymeshb {
namespace su2 {

std::map<int, ElementTypeInfo> get_element_type_map() {
    std::map<int, ElementTypeInfo> map;
    map[3] = {"Edges", 2};            // Line
    map[5] = {"Triangles", 3};        // Triangle
    map[9] = {"Quadrilaterals", 4};   // Quadrilateral
    map[10] = {"Tetrahedra", 4};      // Tetrahedron
    map[12] = {"Prisms", 6};          // Prism
    map[13] = {"Pyramids", 5};        // Pyramid
    map[14] = {"Hexahedra", 8};       // Hexahedron
    return map;
}

std::map<std::string, int> get_reverse_element_type_map() {
    std::map<std::string, int> map;
    map["Edges"] = 3;              // Line
    map["Triangles"] = 5;          // Triangle
    map["Quadrilaterals"] = 9;     // Quadrilateral
    map["Tetrahedra"] = 10;        // Tetrahedron
    map["Prisms"] = 12;            // Prism
    map["Pyramids"] = 13;          // Pyramid
    map["Hexahedra"] = 14;         // Hexahedron
    return map;
}

bool read_elements(std::ifstream& file_stream, int elem_count, py::dict& elements) {
    if (elem_count <= 0) {
        return false;
    }

    // Get element type map
    auto element_type_map = get_element_type_map();

    // First pass: count elements of each type
    std::map<int, int> element_counts;

    // Store the current position to return to it later
    auto initial_position = file_stream.tellg();

    std::string line;
    for (int i = 0; i < elem_count && std::getline(file_stream, line); i++) {
        if (line.empty() || line[0] == '%') {
            // Skip empty lines or comments
            i--;
            continue;
        }

        std::istringstream iss(line);
        int elem_type;
        iss >> elem_type;

        if (element_type_map.find(elem_type) != element_type_map.end()) {
            element_counts[elem_type]++;
        }
    }

    // Clear error flags and reset file position to start of element section
    file_stream.clear();
    file_stream.seekg(initial_position);

    // Create arrays for each element type
    std::map<std::string, py::array_t<unsigned int>> element_arrays;
    std::map<std::string, unsigned int*> element_ptrs;
    std::map<std::string, int> element_indices;

    for (const auto& [elem_type, count] : element_counts) {
        const auto& info = element_type_map[elem_type];
        element_arrays[info.name] = py::array_t<unsigned int>(
            std::vector<py::ssize_t>{count, info.num_node + 1});
        element_ptrs[info.name] = element_arrays[info.name].mutable_data();
        element_indices[info.name] = 0;
    }

    // Second pass: read elements
    for (int i = 0; i < elem_count && std::getline(file_stream, line); i++) {
        if (line.empty() || line[0] == '%') {
            // Skip empty lines or comments
            i--;
            continue;
        }

        std::istringstream iss(line);
        int elem_type;
        iss >> elem_type;

        if (element_type_map.find(elem_type) == element_type_map.end()) {
            continue;
        }

        const auto& info = element_type_map[elem_type];
        int elem_index = element_indices[info.name]++;
        unsigned int* elem_ptr = element_ptrs[info.name] + elem_index * (info.num_node + 1);

        // Read node indices
        for (int j = 0; j < info.num_node; j++) {
            iss >> elem_ptr[j];
        }

        // Read element ID (reference value), default to 0
        int elem_id = 0;
        iss >> elem_id;
        elem_ptr[info.num_node] = elem_id;
    }

    // Create Python dictionary of elements
    for (auto& [name, array] : element_arrays) {
        elements[name.c_str()] = array;
    }

    return true;
}

bool read_boundary_elements(
    std::ifstream& file_stream,
    int boundary_count,
    py::dict& boundaries) {

    if (boundary_count <= 0) {
        return true;
    }

    std::string line;
    auto element_type_map = get_element_type_map();

    // Process each boundary marker
    for (int marker_idx = 0; marker_idx < boundary_count; marker_idx++) {
        // Find marker tag
        std::string marker_tag;
        while (std::getline(file_stream, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }
            if (line.find("MARKER_TAG=") != std::string::npos) {
                marker_tag = pymeshb::trim(line.substr(line.find("=") + 1));
                break;
            }
        }

        // Find marker elements count
        int marker_elements = 0;
        while (std::getline(file_stream, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }
            if (line.find("MARKER_ELEMS=") != std::string::npos) {
                std::istringstream iss(line.substr(line.find("=") + 1));
                iss >> marker_elements;
                break;
            }
        }

        if (marker_elements > 0) {
            // Store boundary elements
            std::vector<std::vector<unsigned int>> edge_elements;
            std::vector<std::vector<unsigned int>> triangle_elements;

            for (int i = 0; i < marker_elements && std::getline(file_stream, line); i++) {
                if (line.empty() || line[0] == '%') {
                    // Skip empty lines or comments
                    i--;
                    continue;
                }

                std::istringstream iss(line);
                int elem_type;
                iss >> elem_type;

                if (elem_type == 3) {
                    // Line
                    std::vector<unsigned int> edge(3);
                    iss >> edge[0] >> edge[1];
                    edge[2] = marker_idx + 1;
                    edge_elements.push_back(edge);
                }
                else if (elem_type == 5) {
                    // Triangle
                    std::vector<unsigned int> tri(4);
                    iss >> tri[0] >> tri[1] >> tri[2];
                    tri[3] = marker_idx + 1;
                    triangle_elements.push_back(tri);
                }
            }

            // Add edge elements to array
            if (!edge_elements.empty()) {
                int num_existing = 0;
                py::array_t<unsigned int> existing_edges;

                // Check if we already have edges
                if (boundaries.contains("Edges")) {
                    existing_edges = boundaries["Edges"].cast<py::array_t<unsigned int>>();
                    num_existing = existing_edges.shape(0);
                }

                // Create new array with combined size
                py::array_t<unsigned int> new_edges(std::vector<py::ssize_t>{num_existing + edge_elements.size(), 3});
                unsigned int* new_edge_ptr = new_edges.mutable_data();

                // Copy existing data if any
                if (num_existing > 0) {
                    unsigned int* old_edge_ptr = existing_edges.mutable_data();
                    for (int i = 0; i < num_existing * 3; i++) {
                        new_edge_ptr[i] = old_edge_ptr[i];
                    }
                }

                // Add new edge elements
                for (size_t i = 0; i < edge_elements.size(); i++) {
                    for (size_t j = 0; j < 3; j++) {
                        new_edge_ptr[(num_existing + i) * 3 + j] = edge_elements[i][j];
                    }
                }

                boundaries["Edges"] = new_edges;
            }

            // Add triangle elements to array (similar process)
            if (!triangle_elements.empty()) {
                int num_existing = 0;
                py::array_t<unsigned int> existing_tris;

                if (boundaries.contains("Triangles")) {
                    existing_tris = boundaries["Triangles"].cast<py::array_t<unsigned int>>();
                    num_existing = existing_tris.shape(0);
                }

                py::array_t<unsigned int> new_tris(std::vector<py::ssize_t>{num_existing + triangle_elements.size(), 4});
                unsigned int* new_tri_ptr = new_tris.mutable_data();

                if (num_existing > 0) {
                    unsigned int* old_tri_ptr = existing_tris.mutable_data();
                    for (int i = 0; i < num_existing * 4; i++) {
                        new_tri_ptr[i] = old_tri_ptr[i];
                    }
                }

                for (size_t i = 0; i < triangle_elements.size(); i++) {
                    for (size_t j = 0; j < 4; j++) {
                        new_tri_ptr[(num_existing + i) * 4 + j] = triangle_elements[i][j];
                    }
                }

                boundaries["Triangles"] = new_tris;
            }
        }
    }

    return true;
}

int process_elements_for_writing(
    const py::dict& elements,
    const py::dict& boundaries,
    int dim,
    std::map<std::string, std::vector<std::vector<unsigned int>>>& interior_elements,
    std::map<int, std::vector<std::vector<unsigned int>>>& boundary_elements) {

    int total_elements = 0;
    std::map<std::string, int> elem_type_map = get_reverse_element_type_map();

    // Process domain elements
    for (auto item : elements) {
        std::string key = item.first.cast<std::string>();
        if (elem_type_map.find(key) == elem_type_map.end()) {
            continue;
        }

        py::array_t<unsigned int> element_array = item.second.cast<py::array_t<unsigned int>>();
        auto elem_ptr = element_array.data();
        int num_elem = element_array.shape(0);
        int num_node = element_array.shape(1) - 1;

        // If these are edges in a 2D/3D mesh, check references to identify boundaries
        if (key == "Edges" && dim > 1) {
            for (int i = 0; i < num_elem; i++) {
                std::vector<unsigned int> elem;
                int ref = elem_ptr[i * (num_node + 1) + num_node];

                // Add nodes
                for (int j = 0; j < num_node; j++) {
                    elem.push_back(elem_ptr[i * (num_node + 1) + j]);
                }

                // If reference > 0, it's a boundary
                if (ref > 0) {
                    boundary_elements[ref].push_back(elem);
                } else {
                    interior_elements[key].push_back(elem);
                }
            }
        } else {
            // For other element types, treat all as interior
            for (int i = 0; i < num_elem; i++) {
                std::vector<unsigned int> elem;
                // Add nodes
                for (int j = 0; j < num_node; j++) {
                    elem.push_back(elem_ptr[i * (num_node + 1) + j]);
                }
                // Add reference
                elem.push_back(elem_ptr[i * (num_node + 1) + num_node]);
                interior_elements[key].push_back(elem);
            }
        }
    }

    // Process boundary elements (if any)
    for (auto item : boundaries) {
        std::string marker_tag = item.first.cast<std::string>();
        int marker_id = 0;

        // Try to extract numeric marker ID from marker tag (assuming format "MARKER_X")
        if (marker_tag.find("MARKER_") == 0) {
             // Skip "MARKER_"
            std::string id_str = marker_tag.substr(7);
            try {
                marker_id = std::stoi(id_str);
            } catch (...) {
                // If not a number, generate sequential marker ID
                marker_id = boundary_elements.size() + 1;
            }
        } else {
            // If marker tag doesn't follow expected format, generate sequential marker ID
            marker_id = boundary_elements.size() + 1;
        }

        py::array_t<unsigned int> boundary_array = item.second.cast<py::array_t<unsigned int>>();
        auto boundary_ptr = boundary_array.data();
        int num_boundary = boundary_array.shape(0);
        int num_node = boundary_array.shape(1) - 1;

        for (int i = 0; i < num_boundary; i++) {
            std::vector<unsigned int> elem;
            // Add nodes
            for (int j = 0; j < num_node; j++) {
                elem.push_back(boundary_ptr[i * (num_node + 1) + j]);
            }
            boundary_elements[marker_id].push_back(elem);
        }
    }

    // Count total elements
    for (const auto& [key, elems] : interior_elements) {
        total_elements += elems.size();
    }

    return total_elements;
}

int write_elements(
    std::ofstream& mesh_file,
    const std::map<std::string, std::vector<std::vector<unsigned int>>>& interior_elements,
    const std::map<std::string, int>& elem_type_map) {

    int elem_id = 0;
    for (const auto& [key, elems] : interior_elements) {
        int vtk_type = elem_type_map.at(key);
        for (const auto& elem : elems) {
            mesh_file << vtk_type << "\t";
            for (size_t i = 0; i < elem.size() - 1; i++) {
                mesh_file << elem[i] << "\t";
            }
            mesh_file << elem_id << std::endl;
            elem_id++;
        }
    }

    return elem_id;
}

bool write_boundary_elements(
    std::ofstream& mesh_file,
    const std::map<int, std::vector<std::vector<unsigned int>>>& boundary_elements) {

    if (boundary_elements.empty()) {
        return true;
    }

    for (const auto& [marker_id, elems] : boundary_elements) {
        mesh_file << "MARKER_TAG= MARKER_" << marker_id << std::endl;
        mesh_file << "MARKER_ELEMS= " << elems.size() << std::endl;

        for (const auto& elem : elems) {
            // Determine element type based on number of nodes
            int vtk_type = (elem.size() == 2) ? 3 :
                          (elem.size() == 3) ? 5 :
                          (elem.size() == 4) ? 9 : 3;

            mesh_file << vtk_type << "\t";
            for (size_t i = 0; i < elem.size(); i++) {
                mesh_file << elem[i] << "\t";
            }
            mesh_file << std::endl;
        }
    }

    return true;
}

} // namespace su2
} // namespace pymeshb