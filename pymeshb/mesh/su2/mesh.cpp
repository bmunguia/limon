#include "mesh.hpp"
#include "solution.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

#include "../util.hpp"

namespace pymeshb {
namespace su2 {

py::tuple read_mesh(const std::string& meshpath, const std::string& solpath, bool read_sol) {
    // Check if file exists
    std::ifstream mesh_file(meshpath);
    if (!mesh_file.is_open()) {
        throw std::runtime_error("Failed to open mesh: " + meshpath);
    }

    // Read the whole file into memory
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(mesh_file, line)) {
        if (line.empty() || line[0] == '%') {  // Skip comments and empty lines
            continue;
        }
        lines.push_back(line);
    }
    mesh_file.close();

    // Parse dimension
    int dim = 2;  // Default
    for (const auto& line : lines) {
        if (line.find("NDIME=") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("=") + 1));
            iss >> dim;
            break;
        }
    }

    // Parse vertices
    int num_ver = 0;
    size_t vertex_section_start = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("NPOIN=") != std::string::npos) {
            std::istringstream iss(lines[i].substr(lines[i].find("=") + 1));
            iss >> num_ver;
            vertex_section_start = i + 1;
            break;
        }
    }

    if (num_ver == 0 || vertex_section_start == 0) {
        throw std::runtime_error("No vertices found in the mesh");
    }

    // Read vertex coordinates
    py::array_t<double> coords(std::vector<py::ssize_t>{num_ver, dim});
    auto coords_ptr = coords.mutable_data();

    for (size_t i = 0; i < num_ver && (vertex_section_start + i) < lines.size(); i++) {
        std::istringstream iss(lines[vertex_section_start + i]);
        for (int j = 0; j < dim; j++) {
            iss >> coords_ptr[i * dim + j];
        }
        // Skip node index
        int node_idx;
        iss >> node_idx;
    }

    // Parse elements
    int elem_count = 0;
    size_t elem_section_start = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("NELEM=") != std::string::npos) {
            std::istringstream iss(lines[i].substr(lines[i].find("=") + 1));
            iss >> elem_count;
            elem_section_start = i + 1;
            break;
        }
    }

    if (elem_count == 0 || elem_section_start == 0) {
        throw std::runtime_error("No elements found in the mesh");
    }

    // Map SU2 VTK types to our element types and number of nodes
    std::map<int, std::pair<std::string, int>> element_type_map = {
        {3, std::make_pair("Edges", 2)},            // Line
        {5, std::make_pair("Triangles", 3)},        // Triangle
        {9, std::make_pair("Quadrilaterals", 4)},   // Quadrilateral
        {10, std::make_pair("Tetrahedra", 4)},      // Tetrahedron
        {12, std::make_pair("Prisms", 6)},          // Prism
        {13, std::make_pair("Pyramids", 5)},        // Pyramid
        {14, std::make_pair("Hexahedra", 8)}        // Hexahedron
    };

    // Count elements of each type to pre-allocate arrays
    std::map<int, int> element_counts;
    for (size_t i = 0; i < elem_count && (elem_section_start + i) < lines.size(); i++) {
        std::istringstream iss(lines[elem_section_start + i]);
        int elem_type;
        iss >> elem_type;
        if (element_type_map.find(elem_type) != element_type_map.end()) {
            element_counts[elem_type]++;
        }
    }

    // Create arrays for each element type
    std::map<std::string, py::array_t<unsigned int>> element_arrays;
    std::map<std::string, unsigned int*> element_ptrs;
    std::map<std::string, int> element_indices;

    for (const auto& [elem_type, count] : element_counts) {
        auto [elem_name, num_nodes] = element_type_map[elem_type];
        element_arrays[elem_name] = py::array_t<unsigned int>(std::vector<py::ssize_t>{count, num_nodes + 1});
        element_ptrs[elem_name] = element_arrays[elem_name].mutable_data();
        element_indices[elem_name] = 0;
    }

    // Read elements
    for (size_t i = 0; i < elem_count && (elem_section_start + i) < lines.size(); i++) {
        std::istringstream iss(lines[elem_section_start + i]);
        int elem_type;
        iss >> elem_type;

        if (element_type_map.find(elem_type) == element_type_map.end()) {
            continue;  // Skip unknown element types
        }

        auto [elem_name, num_nodes] = element_type_map[elem_type];
        int elem_index = element_indices[elem_name]++;
        unsigned int* elem_ptr = element_ptrs[elem_name] + elem_index * (num_nodes + 1);

        // Read node indices
        for (int j = 0; j < num_nodes; j++) {
            iss >> elem_ptr[j];
        }

        // Read element ID (reference value), default to 0
        int elem_id = 0;
        iss >> elem_id;
        elem_ptr[num_nodes] = elem_id;
    }

    // Parse boundary conditions
    int boundary_count = 0;
    size_t boundary_section_start = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("NMARK=") != std::string::npos) {
            std::istringstream iss(lines[i].substr(lines[i].find("=") + 1));
            iss >> boundary_count;
            boundary_section_start = i + 1;
            break;
        }
    }

    if (boundary_count > 0 && boundary_section_start > 0) {
        // Process each boundary marker
        size_t current_line = boundary_section_start;
        for (int marker_idx = 0; marker_idx < boundary_count && current_line < lines.size(); marker_idx++) {
            // Find marker tag
            std::string marker_tag;
            while (current_line < lines.size()) {
                if (lines[current_line].find("MARKER_TAG=") != std::string::npos) {
                    marker_tag = pymeshb::trim(lines[current_line].substr(lines[current_line].find("=") + 1));
                    current_line++;
                    break;
                }
                current_line++;
            }

            // Find marker elements count
            int marker_elements = 0;
            while (current_line < lines.size()) {
                if (lines[current_line].find("MARKER_ELEMS=") != std::string::npos) {
                    std::istringstream iss(lines[current_line].substr(lines[current_line].find("=") + 1));
                    iss >> marker_elements;
                    current_line++;
                    break;
                }
                current_line++;
            }

            if (marker_elements > 0) {
                // Store boundary elements
                std::vector<std::vector<unsigned int>> edge_elements;
                std::vector<std::vector<unsigned int>> triangle_elements;

                for (int i = 0; i < marker_elements && current_line < lines.size(); i++) {
                    std::istringstream iss(lines[current_line++]);
                    int elem_type;
                    iss >> elem_type;

                    if (elem_type == 3) {  // Line
                        std::vector<unsigned int> edge(3);  // 2 nodes + ref
                        iss >> edge[0] >> edge[1];
                        edge[2] = marker_idx + 1;  // Use marker index as reference
                        edge_elements.push_back(edge);
                    }
                    else if (elem_type == 5) {  // Triangle
                        std::vector<unsigned int> tri(4);  // 3 nodes + ref
                        iss >> tri[0] >> tri[1] >> tri[2];
                        tri[3] = marker_idx + 1;
                        triangle_elements.push_back(tri);
                    }
                }

                // Add edge elements to array
                if (!edge_elements.empty()) {
                    int num_existing = 0;
                    if (element_arrays.find("Edges") != element_arrays.end()) {
                        num_existing = element_arrays["Edges"].shape(0);
                    }

                    // Create new array with combined size
                    py::array_t<unsigned int> new_edges(std::vector<py::ssize_t>{num_existing + edge_elements.size(), 3});
                    unsigned int* new_edge_ptr = new_edges.mutable_data();

                    // Copy existing data if any
                    if (num_existing > 0) {
                        unsigned int* old_edge_ptr = element_ptrs["Edges"];
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

                    element_arrays["Edges"] = new_edges;
                    element_ptrs["Edges"] = new_edges.mutable_data();
                }

                // Add triangle elements to array (similar process)
                if (!triangle_elements.empty()) {
                    int num_existing = 0;
                    if (element_arrays.find("Triangles") != element_arrays.end()) {
                        num_existing = element_arrays["Triangles"].shape(0);
                    }

                    py::array_t<unsigned int> new_tris(std::vector<py::ssize_t>{num_existing + triangle_elements.size(), 4});
                    unsigned int* new_tri_ptr = new_tris.mutable_data();

                    if (num_existing > 0) {
                        unsigned int* old_tri_ptr = element_ptrs["Triangles"];
                        for (int i = 0; i < num_existing * 4; i++) {
                            new_tri_ptr[i] = old_tri_ptr[i];
                        }
                    }

                    for (size_t i = 0; i < triangle_elements.size(); i++) {
                        for (size_t j = 0; j < 4; j++) {
                            new_tri_ptr[(num_existing + i) * 4 + j] = triangle_elements[i][j];
                        }
                    }

                    element_arrays["Triangles"] = new_tris;
                    element_ptrs["Triangles"] = new_tris.mutable_data();
                }
            }
        }
    }

    // Create Python dictionary of elements
    py::dict elements;
    for (auto& [name, array] : element_arrays) {
        elements[name.c_str()] = array;
    }

    // Read solution if requested
    py::dict sol;
    if (read_sol && !solpath.empty()) {
        sol = read_solution(solpath, num_ver, dim);
    }

    // Return coords, elements and solution if available
    if (read_sol && sol.size() > 0) {
        return py::make_tuple(coords, elements, sol);
    } else {
        return py::make_tuple(coords, elements);
    }
}

bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                py::dict elements, const std::string& solpath, py::dict sol) {
    // Get dimensions
    int dim = coords.shape(1);
    int64_t num_ver = coords.shape(0);

    // Create output directory if needed
    if (!pymeshb::createDirectory(meshpath)) {
        throw std::runtime_error("Failed to create directory for mesh file");
    }

    // Map from our element types to SU2 VTK types
    std::map<std::string, int> elem_type_map = {
        {"Edges", 3},              // Line
        {"Triangles", 5},          // Triangle
        {"Quadrilaterals", 9},     // Quadrilateral
        {"Tetrahedra", 10},        // Tetrahedron
        {"Prisms", 12},            // Prism
        {"Pyramids", 13},          // Pyramid
        {"Hexahedra", 14}          // Hexahedron
    };

    // Identify boundary elements (edges with reference > 0) and interior elements
    std::map<int, std::vector<std::vector<unsigned int>>> boundary_elements;
    std::map<std::string, std::vector<std::vector<unsigned int>>> interior_elements;

    for (auto item : elements) {
        std::string key = item.first.cast<std::string>();
        if (elem_type_map.find(key) == elem_type_map.end()) {
            continue;  // Skip unknown element types
        }

        py::array_t<unsigned int> element_array = item.second.cast<py::array_t<unsigned int>>();
        auto elem_ptr = element_array.data();
        int64_t num_elem = element_array.shape(0);
        int num_nodes = element_array.shape(1) - 1;

        // If these are edges in a 2D/3D mesh, check references to identify boundaries
        if (key == "Edges" && dim > 1) {
            for (int i = 0; i < num_elem; i++) {
                std::vector<unsigned int> elem;
                int ref = elem_ptr[i * (num_nodes + 1) + num_nodes];

                // Add nodes
                for (int j = 0; j < num_nodes; j++) {
                    elem.push_back(elem_ptr[i * (num_nodes + 1) + j]);
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
                for (int j = 0; j < num_nodes; j++) {
                    elem.push_back(elem_ptr[i * (num_nodes + 1) + j]);
                }
                // Add reference
                elem.push_back(elem_ptr[i * (num_nodes + 1) + num_nodes]);
                interior_elements[key].push_back(elem);
            }
        }
    }

    // Count total elements
    int total_elements = 0;
    for (const auto& [key, elems] : interior_elements) {
        total_elements += elems.size();
    }

    // Write the mesh
    std::ofstream mesh_file(meshpath);
    if (!mesh_file.is_open()) {
        throw std::runtime_error("Failed to open mesh file for writing: " + meshpath);
    }

    // Write dimension
    mesh_file << "% SU2 mesh file generated by pymeshb" << std::endl;
    mesh_file << "%" << std::endl;
    mesh_file << "% Problem dimension" << std::endl;
    mesh_file << "%" << std::endl;
    mesh_file << "NDIME= " << dim << std::endl;

    // Write elements
    mesh_file << "%" << std::endl;
    mesh_file << "% Inner element connectivity" << std::endl;
    mesh_file << "%" << std::endl;
    mesh_file << "NELEM= " << total_elements << std::endl;

    int elem_id = 0;
    for (const auto& [key, elems] : interior_elements) {
        int vtk_type = elem_type_map[key];
        for (const auto& elem : elems) {
            mesh_file << vtk_type << " ";
            for (size_t i = 0; i < elem.size() - 1; i++) {
                mesh_file << elem[i] << " ";
            }
            mesh_file << elem_id << std::endl;
            elem_id++;
        }
    }

    // Write vertices
    auto coords_ptr = coords.data();
    mesh_file << "%" << std::endl;
    mesh_file << "% Node coordinates" << std::endl;
    mesh_file << "%" << std::endl;
    mesh_file << "NPOIN= " << num_ver << std::endl;

    for (int i = 0; i < num_ver; i++) {
        for (int j = 0; j < dim; j++) {
            mesh_file << coords_ptr[i * dim + j] << " ";
        }
        mesh_file << i << std::endl;
    }

    // Write boundary markers
    if (!boundary_elements.empty()) {
        mesh_file << "%" << std::endl;
        mesh_file << "% Boundary elements" << std::endl;
        mesh_file << "%" << std::endl;
        mesh_file << "NMARK= " << boundary_elements.size() << std::endl;

        for (const auto& [marker_id, elems] : boundary_elements) {
            mesh_file << "MARKER_TAG= MARKER_" << marker_id << std::endl;
            mesh_file << "MARKER_ELEMS= " << elems.size() << std::endl;

            for (const auto& elem : elems) {
                // Determine element type
                int vtk_type = (elem.size() == 2) ? 3 : 5; // Line or Triangle
                mesh_file << vtk_type << " ";
                for (size_t i = 0; i < elem.size(); i++) {
                    mesh_file << elem[i] << " ";
                }
                mesh_file << std::endl;
            }
        }
    }

    mesh_file.close();

    // Write solution if provided
    if (!sol.empty() && !solpath.empty()) {
        return write_solution(solpath, sol, num_ver, dim);
    }

    return true;
}

}  // namespace su2
}  // namespace pymeshb