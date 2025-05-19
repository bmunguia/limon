#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>

#include "../util.hpp"
#include "element.hpp"
#include "mesh.hpp"
#include "solution.hpp"

namespace pymeshb {
namespace su2 {

py::tuple read_mesh(const std::string& meshpath, const std::string& solpath,
                    bool read_sol) {
    // Check if file exists
    std::ifstream mesh_file(meshpath);
    if (!mesh_file.is_open()) {
        throw std::runtime_error("Failed to open mesh: " + meshpath);
    }

    // Parse dimension
    int dim = 2;
    std::string line;
    while (std::getline(mesh_file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.find("NDIME=") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("=") + 1));
            iss >> dim;
            break;
        }
    }

    // Elements
    int elem_count = 0;
    while (std::getline(mesh_file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.find("NELEM=") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("=") + 1));
            iss >> elem_count;
            break;
        }
    }

    if (elem_count == 0) {
        throw std::runtime_error("No elements found in the mesh");
    }

    // Read elements
    py::dict elements;
    if (!read_elements(mesh_file, elem_count, elements)) {
        throw std::runtime_error("Failed to read elements from the mesh");
    }

    // Vertices
    int num_ver = 0;
    while (std::getline(mesh_file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.find("NPOIN=") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("=") + 1));
            iss >> num_ver;
            break;
        }
    }

    if (num_ver == 0) {
        throw std::runtime_error("No vertices found in the mesh");
    }

    // Vertex coordinates
    py::array_t<double> coords(std::vector<py::ssize_t>{num_ver, dim});
    auto coords_ptr = coords.mutable_data();

    for (int i = 0; i < num_ver && std::getline(mesh_file, line); i++) {
        if (line.empty() || line[0] == '%') {
            // Skip empty lines or comments
            i--;
            continue;
        }

        std::istringstream iss(line);
        for (int j = 0; j < dim; j++) {
            iss >> coords_ptr[i * dim + j];
        }
        // Skip node index
        int node_idx;
        iss >> node_idx;
    }

    // Parse boundary conditions
    int boundary_count = 0;
    while (std::getline(mesh_file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.find("NMARK=") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("=") + 1));
            iss >> boundary_count;
            break;
        }
    }

    // Read boundary elements
    py::dict boundaries;
    if (boundary_count > 0) {
        if (!read_boundary_elements(mesh_file, boundary_count, boundaries)) {
            throw std::runtime_error("Failed to read boundary elements from the mesh");
        }
    }

    // Read solution if requested
    py::dict sol;
    if (read_sol && !solpath.empty()) {
        sol = read_solution(solpath, num_ver, dim);
    }

    // Return coords, elements and solution if available
    if (read_sol && sol.size() > 0) {
        return py::make_tuple(coords, elements, boundaries, sol);
    } else {
        return py::make_tuple(coords, elements, boundaries);
    }
}

bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                const py::dict& elements, const py::dict& boundaries,
                const std::string& solpath, py::dict sol) {
    // Get dimensions
    int dim = coords.shape(1);
    int num_ver = coords.shape(0);

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

    // Count total elements
    int total_elements = 0;
    for (const auto& [key, elems] : interior_elements) {
        total_elements += elems.size();
    }

    // Write the mesh
    std::ofstream mesh_file;
    mesh_file.open(meshpath);
    mesh_file.precision(15);
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
            mesh_file << vtk_type << "\t";
            for (size_t i = 0; i < elem.size() - 1; i++) {
                mesh_file << elem[i] << "\t";
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
            mesh_file << std::fixed << coords_ptr[i * dim + j] << "\t";
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
                mesh_file << vtk_type << "\t";
                for (size_t i = 0; i < elem.size(); i++) {
                    mesh_file << elem[i] << "\t";
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