#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}

#include "../../common/include/ref_map.hpp"
#include "../../common/include/util.hpp"
#include "../include/element.hpp"
#include "../include/mesh.hpp"

namespace limon {
namespace gmf {

py::tuple load_mesh(const std::string& meshpath, py::dict marker_map,
                    bool read_markers, const std::string& markerpath) {
    int version;
    int dim;

    // Load or initialize marker map
    if (read_markers || marker_map.empty()) {
        marker_map.clear();
        if (!markerpath.empty()) {
            auto cpp_map = RefMap::loadRefMap(markerpath);
            for (const auto& pair : cpp_map) {
                marker_map[py::int_(pair.first)] = py::str(pair.second);
            }
        }
    }

    // Open the mesh
    int64_t mesh_id = GmfOpenMesh(meshpath.c_str(), GmfRead, &version, &dim);
    if (mesh_id == 0) {
        throw std::runtime_error("Failed to open mesh: " + meshpath);
    }

    // Get number of vertices
    int64_t num_ver = GmfStatKwd(mesh_id, GmfVertices);
    if (num_ver == 0) {
        GmfCloseMesh(mesh_id);
        throw std::runtime_error("No vertices found in the mesh");
    }

    //Coordinates array
    py::array_t<double> coords(std::vector<py::ssize_t>{num_ver, dim});
    auto coords_ptr = coords.mutable_data();

    // File position of vertices
    if (!GmfGotoKwd(mesh_id, GmfVertices)) {
        GmfCloseMesh(mesh_id);
        throw std::runtime_error("Failed to find vertices in the mesh");
    }

    // Read vertices
    for (auto i = 0; i < num_ver; i++) {
        int ref;
        if (dim == 2) {
            GmfGetLin(mesh_id, GmfVertices, &coords_ptr[i * dim],
                      &coords_ptr[i * dim + 1], &ref);
        } else {
            GmfGetLin(mesh_id, GmfVertices, &coords_ptr[i * dim],
                      &coords_ptr[i * dim + 1], &coords_ptr[i * dim + 2], &ref);
        }
    }

    // Read elements
    py::dict elements;
    py::dict boundaries;
    if (dim == 2) {
        read_elements_2D(mesh_id, elements, boundaries);
    } else {
        read_elements_3D(mesh_id, elements, boundaries);
    }

    // Close the mesh
    GmfCloseMesh(mesh_id);

    // Return mesh data as a dictionary
    py::dict mesh_data;
    mesh_data["coords"] = coords;
    mesh_data["elements"] = elements;
    mesh_data["boundaries"] = boundaries;
    mesh_data["dim"] = dim;
    mesh_data["num_point"] = static_cast<int>(num_ver);

    return py::make_tuple(mesh_data, marker_map);
}

bool write_mesh(const std::string& meshpath, const py::dict& mesh_data) {
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

    // Get dimensions
    int dim = coords.shape(1);
    int64_t num_ver = coords.shape(0);

    // Create output directory if needed
    if (!limon::createDirectory(meshpath)) {
        throw std::runtime_error("Failed to create directory for mesh file");
    }

    // Open the mesh
    int version = 2;
    int64_t mesh_id = GmfOpenMesh(meshpath.c_str(), GmfWrite, version, dim);
    if (mesh_id == 0) {
        throw std::runtime_error("Failed to open mesh for writing: " + meshpath);
    }

    // Pointer to coordinate data
    auto coords_ptr = coords.data();

    // Set number of vertices
    if (!GmfSetKwd(mesh_id, GmfVertices, num_ver)) {
        GmfCloseMesh(mesh_id);
        throw std::runtime_error("Failed to set vertices keyword");
    }

    // Write vertices
    for (auto i = 0; i < num_ver; i++) {
        if (dim == 2) {
            GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim],
                      coords_ptr[i * dim + 1], 0);
        } else {
            GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim],
                      coords_ptr[i * dim + 1], coords_ptr[i * dim + 2], 0);
        }
    }

    // Write elements
    if (dim == 2) {
        write_elements_2D(mesh_id, elements, boundaries);
    } else {
        write_elements_3D(mesh_id, elements, boundaries);
    }

    // Close the mesh
    GmfCloseMesh(mesh_id);

    return true;
}

} // namespace gmf
} // namespace limon