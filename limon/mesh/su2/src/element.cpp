#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "../../common/include/ref_map.hpp"
#include "../../common/include/util.hpp"
#include "../include/element.hpp"

namespace limon {
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

void read_element_type(std::ifstream& file_stream, int elem_type, int count, py::dict& elements, const std::string& key_name) {
    auto element_type_map = get_element_type_map();
    if (element_type_map.find(elem_type) == element_type_map.end()) {
        throw std::runtime_error("Unknown element type: " + std::to_string(elem_type));
    }

    int num_node = element_type_map[elem_type].num_node;

    // Count actual elements of this type
    auto initial_position = file_stream.tellg();
    std::string line;
    int found_count = 0;

    for (int i = 0; i < count && std::getline(file_stream, line); i++) {
        if (line.empty() || line[0] == '%') {
            // Skip comment or empty line
            i--;
            continue;
        }

        std::istringstream iss(line);
        int line_elem_type;
        iss >> line_elem_type;

        if (line_elem_type == elem_type) {
            found_count++;
        }
    }

    // Reset file position
    file_stream.clear();
    file_stream.seekg(initial_position);

    if (found_count > 0) {
        // Create array for this element type
        py::array_t<unsigned int> element_array(std::vector<py::ssize_t>{found_count, num_node + 1});
        auto elm_ptr = element_array.mutable_data();
        int curr_idx = 0;

        // Read elements
        for (int i = 0; i < count && std::getline(file_stream, line); i++) {
            if (line.empty() || line[0] == '%') {
                // Skip comment or empty line
                i--;
                continue;
            }

            std::istringstream iss(line);
            int line_elem_type;
            iss >> line_elem_type;

            if (line_elem_type == elem_type) {
                // Read node indices
                for (int j = 0; j < num_node; j++) {
                    iss >> elm_ptr[curr_idx * (num_node + 1) + j];
                }

                // Hard-code volume reference ID
                int ref = 0;
                elm_ptr[curr_idx * (num_node + 1) + num_node] = ref;
                curr_idx++;
            }
        }

        // Reset file position
        file_stream.clear();
        file_stream.seekg(initial_position);

        // Add to elements dictionary
        elements[key_name.c_str()] = element_array;
    }
}

void read_elements_2D(std::ifstream& file_stream, int elem_count, py::dict& elements) {
    // Store initial position
    auto initial_position = file_stream.tellg();

    // Read volume elements (2D)
    read_element_type(file_stream, 5, elem_count, elements, "Triangles");
    read_element_type(file_stream, 9, elem_count, elements, "Quadrilaterals");

    // Reset file position
    file_stream.clear();
    file_stream.seekg(initial_position);
}

void read_elements_3D(std::ifstream& file_stream, int elem_count, py::dict& elements) {
    // Store initial position
    auto initial_position = file_stream.tellg();

    // Read volume elements (3D)
    read_element_type(file_stream, 10, elem_count, elements, "Tetrahedra");
    read_element_type(file_stream, 12, elem_count, elements, "Prisms");
    read_element_type(file_stream, 13, elem_count, elements, "Pyramids");
    read_element_type(file_stream, 14, elem_count, elements, "Hexahedra");

    // Reset file position
    file_stream.clear();
    file_stream.seekg(initial_position);
}

void read_boundary_element_type(std::ifstream& file_stream, int marker_idx, int marker_elements,
                               int elem_type, py::dict& boundaries, const std::string& key_name) {
    auto element_type_map = get_element_type_map();
    if (element_type_map.find(elem_type) == element_type_map.end()) {
        // Unknown element type
        return;
    }

    int num_node = element_type_map[elem_type].num_node;

    // Count actual elements of this type
    auto initial_position = file_stream.tellg();
    std::string line;
    int found_count = 0;

    for (int i = 0; i < marker_elements && std::getline(file_stream, line); i++) {
        if (line.empty() || line[0] == '%') {
            // Skip comment or empty line
            i--;
            continue;
        }

        std::istringstream iss(line);
        int line_elem_type;
        iss >> line_elem_type;

        if (line_elem_type == elem_type) {
            found_count++;
        }
    }

    // Reset file position
    file_stream.clear();
    file_stream.seekg(initial_position);

    if (found_count > 0) {
        // Get existing array if it exists
        py::array_t<unsigned int> element_array;
        int existing_count = 0;

        if (boundaries.contains(key_name)) {
            element_array = boundaries[key_name.c_str()].cast<py::array_t<unsigned int>>();
            existing_count = element_array.shape(0);

            // Create new array with extended size
            py::array_t<unsigned int> new_array(std::vector<py::ssize_t>{existing_count + found_count, num_node + 1});
            auto new_ptr = new_array.mutable_data();

            // Copy existing data
            auto existing_ptr = element_array.data();
            for (int i = 0; i < existing_count * (num_node + 1); i++) {
                new_ptr[i] = existing_ptr[i];
            }

            element_array = new_array;
        } else {
            // Create new array
            element_array = py::array_t<unsigned int>(std::vector<py::ssize_t>{found_count, num_node + 1});
        }

        auto elm_ptr = element_array.mutable_data();
        int curr_idx = existing_count;

        // Read boundary elements
        for (int i = 0; i < marker_elements && std::getline(file_stream, line); i++) {
            if (line.empty() || line[0] == '%') {
                // Skip comment or empty line
                i--;
                continue;
            }

            std::istringstream iss(line);
            int line_elem_type;
            iss >> line_elem_type;

            if (line_elem_type == elem_type) {
                // Read node indices
                for (int j = 0; j < num_node; j++) {
                    iss >> elm_ptr[curr_idx * (num_node + 1) + j];
                }

                // Set reference ID to marker index + 1
                elm_ptr[curr_idx * (num_node + 1) + num_node] = marker_idx + 1;
                curr_idx++;
            }
        }

        // Reset file position
        file_stream.clear();
        file_stream.seekg(initial_position);

        // Add to boundaries dictionary
        boundaries[key_name.c_str()] = element_array;
    }
}

void read_boundary_elements_2D(std::ifstream& file_stream, int marker_idx, int marker_elements,
                               py::dict& boundaries) {
    // In 2D, boundaries are primarily edges (type 3)
    read_boundary_element_type(file_stream, marker_idx, marker_elements, 3, boundaries, "Edges");
}

void read_boundary_elements_3D(std::ifstream& file_stream, int marker_idx, int marker_elements,
                               py::dict& boundaries) {
    // In 3D, boundaries can be triangles (type 5) or quadrilaterals (type 9)
    read_boundary_element_type(file_stream, marker_idx, marker_elements, 5, boundaries, "Triangles");
    read_boundary_element_type(file_stream, marker_idx, marker_elements, 9, boundaries, "Quadrilaterals");
}

std::map<int, std::string> read_boundary_elements(std::ifstream& file_stream, int boundary_count, py::dict& boundaries,
                                                  bool write_markers, const std::string& markerpath) {
    // Process each boundary marker
    std::string line;
    std::map<int, std::string> ref_map;
    for (int marker_idx = 0; marker_idx < boundary_count; marker_idx++) {
        // Find marker tag
        std::string marker_tag;
        while (std::getline(file_stream, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }
            if (line.find("MARKER_TAG=") != std::string::npos) {
                marker_tag = line.substr(line.find("=") + 1);
                marker_tag = trim(marker_tag);
                break;
            }
        }

        // Store marker tag in map with index + 1 as key
        if (!marker_tag.empty()) {
            ref_map[marker_idx + 1] = marker_tag;
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
            // Store initial position
            auto initial_position = file_stream.tellg();

            // Try to read as 2D boundary
            read_boundary_elements_2D(file_stream, marker_idx, marker_elements, boundaries);

            // Reset position
            file_stream.clear();
            file_stream.seekg(initial_position);

            // Try to read as 3D boundary
            read_boundary_elements_3D(file_stream, marker_idx, marker_elements, boundaries);

            // Skip through these elements to get to the next marker
            for (int i = 0; i < marker_elements; i++) {
                std::getline(file_stream, line);
                if (line.empty() || line[0] == '%') {
                    // Skip comment or empty line
                    i--;
                }
            }
        }
    }

    // Save updated marker map back to file if provided
    if (write_markers && !markerpath.empty()) {
        RefMap::writeRefMap(ref_map, markerpath, RefMapKind::Marker);
    }

    return ref_map;
}

void write_element_type(std::ofstream& mesh_file, int elem_type, py::array_t<unsigned int>& element_array) {
    auto elem_buf = element_array.request();
    unsigned int* elem_ptr = static_cast<unsigned int*>(elem_buf.ptr);
    int64_t num_elm = element_array.shape(0);
    int num_node = element_array.shape(1) - 1;

    // Write each element
    for (int i = 0; i < num_elm; i++) {
        mesh_file << elem_type << "\t";
        for (int j = 0; j < num_node; j++) {
            mesh_file << elem_ptr[i * (num_node + 1) + j] << "\t";
        }
        mesh_file << i << std::endl;
    }
}

void write_elements_2D(std::ofstream& mesh_file, const py::dict& elements, int& elem_count) {
    auto elem_type_map = get_reverse_element_type_map();

    // Write triangles if present
    if (elements.contains("Triangles")) {
        auto element_array = elements["Triangles"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Triangles"], element_array);
        elem_count += element_array.shape(0);
    }

    // Write quadrilaterals if present
    if (elements.contains("Quadrilaterals")) {
        auto element_array = elements["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Quadrilaterals"], element_array);
        elem_count += element_array.shape(0);
    }
}

void write_elements_3D(std::ofstream& mesh_file, const py::dict& elements, int& elem_count) {
    auto elem_type_map = get_reverse_element_type_map();

    // Write tetrahedra if present
    if (elements.contains("Tetrahedra")) {
        auto element_array = elements["Tetrahedra"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Tetrahedra"], element_array);
        elem_count += element_array.shape(0);
    }

    // Write prisms if present
    if (elements.contains("Prisms")) {
        auto element_array = elements["Prisms"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Prisms"], element_array);
        elem_count += element_array.shape(0);
    }

    // Write pyramids if present
    if (elements.contains("Pyramids")) {
        auto element_array = elements["Pyramids"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Pyramids"], element_array);
        elem_count += element_array.shape(0);
    }

    // Write hexahedra if present
    if (elements.contains("Hexahedra")) {
        auto element_array = elements["Hexahedra"].cast<py::array_t<unsigned int>>();
        write_element_type(mesh_file, elem_type_map["Hexahedra"], element_array);
        elem_count += element_array.shape(0);
    }
}

int write_elements(std::ofstream& mesh_file, const py::dict& elements) {
    int elem_count = 0;

    // Write both 2D and 3D elements
    write_elements_2D(mesh_file, elements, elem_count);
    write_elements_3D(mesh_file, elements, elem_count);

    return elem_count;
}

void write_boundary_elements_2D(std::ofstream& mesh_file, const py::dict& boundaries,
                                std::map<int, std::string>& ref_map) {
    if (!boundaries.contains("Edges")) {
        return;
    }

    auto element_array = boundaries["Edges"].cast<py::array_t<unsigned int>>();
    auto elem_ptr = element_array.data();
    int num_elem = element_array.shape(0);
    int num_node = element_array.shape(1) - 1;

    // Group by reference (marker ID)
    std::map<unsigned int, std::vector<std::vector<unsigned int>>> marker_elements;
    for (int i = 0; i < num_elem; i++) {
        unsigned int ref = elem_ptr[i * (num_node + 1) + num_node];
        std::vector<unsigned int> nodes;
        for (int j = 0; j < num_node; j++) {
            nodes.push_back(elem_ptr[i * (num_node + 1) + j]);
        }
        marker_elements[ref].push_back(nodes);
    }

    // Write marker sections
    for (auto& [marker_id, elements] : marker_elements) {
        // Load marker name or fallback to REF_<marker_id>
        std::string marker_name = RefMap::getRefName(ref_map, marker_id);

        mesh_file << "MARKER_TAG= " << marker_name << std::endl;
        mesh_file << "MARKER_ELEMS= " << elements.size() << std::endl;

        for (auto& nodes : elements) {
            mesh_file << "3\t"; // VTK_LINE
            for (auto node : nodes) {
                mesh_file << node << "\t";
            }
            mesh_file << std::endl;
        }
    }
}

void write_boundary_elements_3D(std::ofstream& mesh_file, const py::dict& boundaries,
                                std::map<int, std::string>& ref_map) {
    std::map<unsigned int, std::vector<std::vector<unsigned int>>> triangles_by_marker;
    std::map<unsigned int, std::vector<std::vector<unsigned int>>> quads_by_marker;

    // Process triangles
    if (boundaries.contains("Triangles")) {
        auto element_array = boundaries["Triangles"].cast<py::array_t<unsigned int>>();
        auto elem_ptr = element_array.data();
        int num_elem = element_array.shape(0);
        int num_node = element_array.shape(1) - 1;

        for (int i = 0; i < num_elem; i++) {
            unsigned int ref = elem_ptr[i * (num_node + 1) + num_node];
            std::vector<unsigned int> nodes;
            for (int j = 0; j < num_node; j++) {
                nodes.push_back(elem_ptr[i * (num_node + 1) + j]);
            }
            triangles_by_marker[ref].push_back(nodes);
        }
    }

    // Process quadrilaterals
    if (boundaries.contains("Quadrilaterals")) {
        auto element_array = boundaries["Quadrilaterals"].cast<py::array_t<unsigned int>>();
        auto elem_ptr = element_array.data();
        int num_elem = element_array.shape(0);
        int num_node = element_array.shape(1) - 1;

        for (int i = 0; i < num_elem; i++) {
            unsigned int ref = elem_ptr[i * (num_node + 1) + num_node];
            std::vector<unsigned int> nodes;
            for (int j = 0; j < num_node; j++) {
                nodes.push_back(elem_ptr[i * (num_node + 1) + j]);
            }
            quads_by_marker[ref].push_back(nodes);
        }
    }

    // Combine markers from both types
    std::set<unsigned int> all_markers;
    for (auto& [marker, _] : triangles_by_marker) all_markers.insert(marker);
    for (auto& [marker, _] : quads_by_marker) all_markers.insert(marker);

    // Write marker sections
    for (unsigned int marker_id : all_markers) {
        // Load marker name or fallback to MARKER_<marker_id>
        std::string marker_name = RefMap::getRefName(ref_map, marker_id);

        // Count total elements for this marker
        size_t total_elements =
            (triangles_by_marker.count(marker_id) ? triangles_by_marker[marker_id].size() : 0) +
            (quads_by_marker.count(marker_id) ? quads_by_marker[marker_id].size() : 0);

        mesh_file << "MARKER_TAG= " << marker_name << std::endl;
        mesh_file << "MARKER_ELEMS= " << total_elements << std::endl;

        // Write triangles for this marker
        if (triangles_by_marker.count(marker_id)) {
            for (auto& nodes : triangles_by_marker[marker_id]) {
                mesh_file << "5\t"; // VTK_TRIANGLE
                for (auto node : nodes) {
                    mesh_file << node << "\t";
                }
                mesh_file << std::endl;
            }
        }

        // Write quadrilaterals for this marker
        if (quads_by_marker.count(marker_id)) {
            for (auto& nodes : quads_by_marker[marker_id]) {
                mesh_file << "9\t"; // VTK_QUAD
                for (auto node : nodes) {
                    mesh_file << node << "\t";
                }
                mesh_file << std::endl;
            }
        }
    }
}

int write_boundary_elements(std::ofstream& mesh_file, const py::dict& boundaries,
                            const std::map<int, std::string>& marker_map) {
    if (boundaries.empty()) {
        return 0;
    }

    // Use provided marker map
    std::map<int, std::string> ref_map = marker_map;

    // Write both 2D and 3D boundaries
    write_boundary_elements_2D(mesh_file, boundaries, ref_map);
    write_boundary_elements_3D(mesh_file, boundaries, ref_map);

    return ref_map.size();
}

} // namespace su2
} // namespace limon