#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
extern "C" {
#include <libmeshb7.h>
}

namespace py = pybind11;

PYBIND11_MODULE(libmeshb, m) {
    m.doc() = "Python bindings for libMeshb";

    m.def("read_mesh", [](const std::string& filepath, bool read_sol = false) {
        int ver;
        int dim;

        // Open mesh for reading
        int64_t mesh_id = GmfOpenMesh(filepath.c_str(), GmfRead, &ver, &dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh: " + filepath);
        }

        // Get number of vertices
        int64_t num_ver = GmfStatKwd(mesh_id, GmfVertices);
        if (num_ver == 0) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("No vertices found in the mesh");
        }

        // Coordinates array
        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(num_ver),
            static_cast<py::ssize_t>(dim)
        };
        py::array_t<double> coords(shape);
        auto coords_ptr = coords.mutable_data();

        // File position of vertices
        if (!GmfGotoKwd(mesh_id, GmfVertices)) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("Failed to find vertices in the mesh");
        }

        // Read vertices
        for (int64_t i = 0; i < num_ver; i++) {
            int ref;
            double x, y, z = 0.0;

            if (dim == 2) {
                GmfGetLin(mesh_id, GmfVertices, &x, &y, &ref);
                coords_ptr[i*dim] = x;
                coords_ptr[i*dim+1] = y;
            } else {
                GmfGetLin(mesh_id, GmfVertices, &x, &y, &z, &ref);
                coords_ptr[i*dim] = x;
                coords_ptr[i*dim+1] = y;
                coords_ptr[i*dim+2] = z;
            }
        }

        // Read solution if requested
        py::dict sol;

        if (read_sol) {
            int num_types, sol_size;
            int types[GmfMaxTyp];

            // Check if solution exist
            int64_t num_lin = GmfStatKwd(mesh_id, GmfSolAtVertices, &num_types, &sol_size, types);

            if (num_lin > 0) {
                // Solution file position
                if (GmfGotoKwd(mesh_id, GmfSolAtVertices)) {
                    // Solution array
                    std::vector<py::array_t<double>> sol_array;
                    py::array_t<double> buffer(sol_size);
                    for (int64_t i = 0; i < num_ver; i++) {
                        GmfGetLin(mesh_id, GmfSolAtVertices, &buffer);
                        sol_array.push_back(buffer);
                    }

                    // Solution dict
                    int count = 0;
                    for (int i = 0; i < num_types; i++) {
                        std::string field_name = "sol_" + std::to_string(i);

                        // Create appropriate array based on field type
                        if (types[i] == GmfSca) {
                            // Scalar field - 1D array
                            py::array_t<double> scalar_field(num_ver);
                            auto scalar_ptr = scalar_field.mutable_data();

                            // Read scalar values
                            for (int64_t j = 0; j < num_ver; j++) {
                                auto sol_data = sol_array[j].data();
                                scalar_ptr[j] = sol_data[count + i];
                            }

                            sol[field_name.c_str()] = scalar_field;
                            count += 1;

                        } else if (types[i] == GmfVec) {
                            // Vector field - 2D array (num_ver x dim)
                            std::vector<py::ssize_t> vec_shape = {
                                static_cast<py::ssize_t>(num_ver),
                                static_cast<py::ssize_t>(dim)
                            };
                            py::array_t<double> vector_field(vec_shape);
                            auto vector_ptr = vector_field.mutable_data();

                            // Read vector values
                            for (int64_t j = 0; j < num_ver; j++) {
                                for (int k = 0; k < dim; k++) {
                                    auto sol_data = sol_array[j].data();
                                    vector_ptr[j*dim + k] = sol_data[count + k];
                                }
                            }

                            sol[field_name.c_str()] = vector_field;
                            count += dim;

                        } else if (types[i] == GmfSymMat) {
                            // Symmetric matrix - 2D array with (dim*(dim+1))/2 components per vertex
                            int sym_size = (dim*(dim+1))/2;
                            std::vector<py::ssize_t> mat_shape = {
                                static_cast<py::ssize_t>(num_ver),
                                static_cast<py::ssize_t>(sym_size)
                            };
                            py::array_t<double> matrix_field(mat_shape);
                            auto matrix_ptr = matrix_field.mutable_data();

                            // Read symmetric matrix values
                            for (int64_t j = 0; j < num_ver; j++) {
                                for (int k = 0; k < sym_size; k++) {
                                    auto sol_data = sol_array[j].data();
                                    matrix_ptr[j*sym_size + k] = sol_data[count + k];
                                }
                            }

                            sol[field_name.c_str()] = matrix_field;
                            count += sym_size;
                        }
                    }
                }
            }
        }

        // Close the mesh
        GmfCloseMesh(mesh_id);

        // Return coords and solution if available
        if (read_sol && sol.size() > 0) {
            return py::make_tuple(coords, sol);
        } else {
            return py::make_tuple(coords);
        }
    }, py::arg("filepath"), py::arg("read_sol") = false,
       "Read a meshb file and return nodes and coordinates as numpy arrays");

       m.def("write_mesh", [](const std::string& filepath, py::array_t<double> coords,
                              py::dict sol = py::dict()) {
        int ver = 4;
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
        int64_t mesh_id = GmfOpenMesh(filepath.c_str(), GmfWrite, ver, dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh for writing: " + filepath);
        }

        // Pointer to coordinate data
        auto coords_ptr = coords.data();

        // Set number of vertices
        int64_t num_ver = coords.shape(0);
        if (!GmfSetKwd(mesh_id, GmfVertices, num_ver)) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("Failed to set vertices keyword");
        }

        // Write vertices
        for (int64_t i = 0; i < num_ver; i++) {
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

        // Write solution if provided
    if (!sol.empty()) {
        std::vector<int> field_type;
        std::vector<py::array_t<double>> field_array;

        for (auto item : sol) {
            py::array_t<double> field = item.second.cast<py::array_t<double>>();

            // Determine field type from array shape
            if (field.ndim() == 1) {
                // Scalar field
                field_type.push_back(GmfSca);
            } else if (field.ndim() == 2 && field.shape(1) == dim) {
                // Vector field
                field_type.push_back(GmfVec);
            } else if (field.ndim() == 2 && field.shape(1) == (dim * (dim + 1)) / 2) {
                // Symmetric matrix field
                field_type.push_back(GmfSymMat);
            } else {
                continue; // Skip fields with unsupported shapes
            }

            field_array.push_back(field);
        }

        // Write solution fields if any valid fields were found
        if (!field_type.empty()) {
            // Set solution keyword with field types
            GmfSetKwd(mesh_id, GmfSolAtVertices, num_ver, field_type.size(), field_type.data());

            // Write solution values for each vertex
            std::vector<double> buffer;
            for (int64_t i = 0; i < num_ver; i++) {
                buffer.clear();

                for (size_t j = 0; j < field_type.size(); j++) {
                    auto field_ptr = field_array[j].data();

                    if (field_type[j] == GmfSca) {
                        // Add scalar value to buffer
                        buffer.push_back(field_ptr[i]);
                    } else if (field_type[j] == GmfVec) {
                        // Add vector values to buffer
                        for (int k = 0; k < dim; k++) {
                            buffer.push_back(field_ptr[i * dim + k]);
                        }
                    } else if (field_type[j] == GmfSymMat) {
                        // Add symmetric matrix values to buffer
                        int sym_size = (dim * (dim + 1)) / 2;
                        for (int k = 0; k < sym_size; k++) {
                            buffer.push_back(field_ptr[i * sym_size + k]);
                        }
                    }
                }

                // Write the buffer for the current vertex
                GmfSetLin(mesh_id, GmfSolAtVertices, buffer.data());
            }
        }
    }

        // Close the mesh
        GmfCloseMesh(mesh_id);
        return true;
    }, py::arg("filepath"), py::arg("coords"), py::arg("sol") = py::dict(),
       "Write nodes and coordinates to a meshb file");
}