#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}

#include "../util.hpp"
#include "mesh.hpp"
#include "solution.hpp"

namespace pymeshb {
namespace gmf {

py::tuple read_mesh(const std::string& meshpath, const std::string& solpath,
                    bool read_sol) {
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
    std::vector<std::pair<int, int>> element_types = {
        {GmfEdges, 2}, {GmfTriangles, 3}, {GmfQuadrilaterals, 4},
        {GmfTetrahedra, 4}, {GmfPrisms, 6}, {GmfHexahedra, 8}
    };

    for (const auto& [kwd, num_nodes] : element_types) {
        int64_t num_elm = GmfStatKwd(mesh_id, kwd);
        if (num_elm > 0) {
            py::array_t<unsigned int> element_array(std::vector<py::ssize_t>{num_elm, num_nodes + 1});
            auto elm_ptr = element_array.mutable_data();

            if (GmfGotoKwd(mesh_id, kwd)) {
                for (auto i = 0; i < num_elm; i++) {
                    std::vector<int> bufInt(num_nodes);
                    int ref;
                    if (kwd == GmfEdges) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &ref);
                    else if (kwd == GmfTriangles) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &ref);
                    else if (kwd == GmfQuadrilaterals) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &ref);
                    else if (kwd == GmfTetrahedra) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &ref);
                    else if (kwd == GmfPrisms) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &bufInt[4], &bufInt[5], &ref);
                    else if (kwd == GmfHexahedra) GmfGetLin(mesh_id, kwd, &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &bufInt[4], &bufInt[5], &bufInt[6], &bufInt[7], &ref);

                    for (auto j = 0; j < num_nodes; j++) {
                        elm_ptr[i * (num_nodes + 1) + j] = bufInt[j];
                    }
                    elm_ptr[i * (num_nodes + 1) + num_nodes] = ref;
                }
            }

            std::string key;
            if (kwd == GmfEdges) key = "Edges";
            else if (kwd == GmfTriangles) key = "Triangles";
            else if (kwd == GmfQuadrilaterals) key = "Quadrilaterals";
            else if (kwd == GmfTetrahedra) key = "Tetrahedra";
            else if (kwd == GmfPrisms) key = "Prisms";
            else if (kwd == GmfHexahedra) key = "Hexahedra";
            else throw std::runtime_error("Unknown keyword in element reading");
            elements[key.c_str()] = element_array;
        }
    }

    // Close the mesh
    GmfCloseMesh(mesh_id);

    py::dict sol_dict;
    if (read_sol && !solpath.empty()) {
        sol_dict = read_solution(solpath, num_ver, dim);
    }

    if (read_sol && sol_dict.size() > 0) {
        return py::make_tuple(coords, elements, sol_dict);
    } else {
        return py::make_tuple(coords, elements);
    }
}

bool write_mesh(const std::string& meshpath, py::array_t<double> coords,
                py::dict elements, const std::string& solpath, py::dict sol) {
    // Get dimensions
    int version = 4;
    int dim = coords.shape(1);
    int64_t num_ver = coords.shape(0);

    // Create output directory if needed
    if (!pymeshb::createDirectory(meshpath)) {
        throw std::runtime_error("Failed to create directory for mesh file");
    }

    // Open the mesh
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
            GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim], coords_ptr[i * dim + 1], 0);
        } else {
            GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim], coords_ptr[i * dim + 1], coords_ptr[i * dim + 2], 0);
        }
    }

    // Write elements
    for (auto item : elements) {
        std::string key = item.first.cast<std::string>();
        py::array_t<unsigned int> element_array = item.second.cast<py::array_t<unsigned int>>();
        auto elm_ptr = element_array.data();
        int64_t num_elm = element_array.shape(0);
        int num_nodes = element_array.shape(1) - 1;

        int kwd;
        if (key == "Edges") kwd = GmfEdges;
        else if (key == "Triangles") kwd = GmfTriangles;
        else if (key == "Quadrilaterals") kwd = GmfQuadrilaterals;
        else if (key == "Tetrahedra") kwd = GmfTetrahedra;
        else if (key == "Prisms") kwd = GmfPrisms;
        else if (key == "Hexahedra") kwd = GmfHexahedra;
        else continue;

        GmfSetKwd(mesh_id, kwd, num_elm);
        for (auto i = 0; i < num_elm; i++) {
            int base_idx = i * (num_nodes + 1);
            if (kwd == GmfEdges) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2]);
            else if (kwd == GmfTriangles) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2], elm_ptr[base_idx+3]);
            else if (kwd == GmfQuadrilaterals) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2], elm_ptr[base_idx+3], elm_ptr[base_idx+4]);
            else if (kwd == GmfTetrahedra) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2], elm_ptr[base_idx+3], elm_ptr[base_idx+4]);
            else if (kwd == GmfPrisms) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2], elm_ptr[base_idx+3], elm_ptr[base_idx+4], elm_ptr[base_idx+5], elm_ptr[base_idx+6]);
            else if (kwd == GmfHexahedra) GmfSetLin(mesh_id, kwd, elm_ptr[base_idx], elm_ptr[base_idx+1], elm_ptr[base_idx+2], elm_ptr[base_idx+3], elm_ptr[base_idx+4], elm_ptr[base_idx+5], elm_ptr[base_idx+6], elm_ptr[base_idx+7], elm_ptr[base_idx+8]);
        }
    }

    // Close the mesh
    GmfCloseMesh(mesh_id);

    // Write solution if provided
    if (!sol.empty() && !solpath.empty()) {
        return write_solution(solpath, sol, num_ver, dim, version);
    }
    return true;
}

} // namespace gmf
} // namespace pymeshb