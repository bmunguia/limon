#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}

#include "../util.hpp"
#include "element.hpp"
#include "mesh.hpp"

namespace pymeshb {
namespace gmf {

py::tuple read_mesh(const std::string& meshpath) {
    int version;
    int dim;

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

    return py::make_tuple(coords, elements, boundaries);
}

bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                const py::dict& elements, const py::dict& boundaries) {
    // Get dimensions
    int dim = coords.shape(1);
    int64_t num_ver = coords.shape(0);

    // Create output directory if needed
    if (!pymeshb::createDirectory(meshpath)) {
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
} // namespace pymeshb