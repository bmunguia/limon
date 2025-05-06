#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
extern "C" {
#include <libmeshb7.h>
}

namespace py = pybind11;

PYBIND11_MODULE(libmeshb, m) {
    m.doc() = "Python bindings for libMeshb";

    m.def("read_mesh", [](const std::string& filepath) {
        int version;
        int dim;

        // Open mesh for reading
        int64_t mesh_id = GmfOpenMesh(filepath.c_str(), GmfRead, &version, &dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh: " + filepath);
        }

        // Get number of vertices
        int64_t num_vertices = GmfStatKwd(mesh_id, GmfVertices);
        if (num_vertices == 0) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("No vertices found in the mesh");
        }

        // Initialize arrays for nodes and coordinates
        py::array_t<int> nodes(num_vertices);
        py::array_t<double> coords(std::vector<ptrdiff_t>{num_vertices, (size_t)dim});

        // Pointers to data
        auto nodes_ptr = nodes.mutable_data();
        auto coords_ptr = coords.mutable_data();

        // File position of vertices
        if (!GmfGotoKwd(mesh_id, GmfVertices)) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("Failed to find vertices in the mesh");
        }

        // Read vertices
        for (int64_t i = 0; i < num_vertices; i++) {
            int node_id, ref;
            double x, y, z = 0.0;

            if (dim == 2) {
                GmfGetLin(mesh_id, GmfVertices, &x, &y, &ref);
                nodes_ptr[i] = i;
                coords_ptr[i*dim] = x;
                coords_ptr[i*dim+1] = y;
            } else {
                GmfGetLin(mesh_id, GmfVertices, &x, &y, &z, &ref);
                nodes_ptr[i] = i;
                coords_ptr[i*dim] = x;
                coords_ptr[i*dim+1] = y;
                coords_ptr[i*dim+2] = z;
            }
        }

        // Close the mesh
        GmfCloseMesh(mesh_id);

        // Return the nodes and coordinates
        return py::make_tuple(nodes, coords);
    }, py::arg("filepath"),
       "Read a meshb file and return nodes and coordinates as numpy arrays");

    m.def("write_mesh", [](py::array_t<int> nodes, py::array_t<double> coords, const std::string& filepath) {
        int version = 4;
        int dim = coords.shape(1);

        // Create output directory if needed
        size_t last_slash = filepath.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string dir = filepath.substr(0, last_slash);
            std::string cmd = "mkdir -p " + dir;
            int ret = system(cmd.c_str());
            if (ret != 0) {
                throw std::runtime_error("Failed to create directory: " + dir);
            }
        }

        // Open mesh for writing
        int64_t mesh_id = GmfOpenMesh(filepath.c_str(), GmfWrite, version, dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh for writing: " + filepath);
        }

        // Pointers to data
        auto nodes_ptr = nodes.data();
        auto coords_ptr = coords.data();

        // Set number of vertices
        int64_t num_vertices = nodes.size();
        if (!GmfSetKwd(mesh_id, GmfVertices, num_vertices)) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("Failed to set vertices keyword");
        }

        // Write vertices
        for (int64_t i = 0; i < num_vertices; i++) {
            int node_id = nodes_ptr[i];

            if (dim == 2) {
                double x = coords_ptr[i*dim];
                double y = coords_ptr[i*dim+1];
                GmfSetLin(mesh_id, GmfVertices, x, y, 0);
            } else {
                double x = coords_ptr[i*dim];
                double y = coords_ptr[i*dim+1];
                double z = coords_ptr[i*dim+2];
                GmfSetLin(mesh_id, GmfVertices, x, y, z, 0);
            }
        }

        // Close the mesh
        GmfCloseMesh(mesh_id);
        return true;
    }, py::arg("nodes"), py::arg("coords"), py::arg("filepath"),
       "Write nodes and coordinates to a meshb file");
}