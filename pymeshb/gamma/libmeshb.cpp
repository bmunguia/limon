#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
extern "C" {
#include <libmeshb7.h>
}

namespace py = pybind11;

PYBIND11_MODULE(libmeshb, m) {
    m.doc() = "Python bindings for libMeshb";

    m.def("read_mesh", [](const std::string& meshpath, const std::string& solpath = "",
                          bool read_sol = false) {
        int ver;
        int dim;

        // Open mesh for reading
        int64_t mesh_id = GmfOpenMesh(meshpath.c_str(), GmfRead, &ver, &dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh: " + meshpath);
        }

        // Get number of vertices
        int64_t num_ver = GmfStatKwd(mesh_id, GmfVertices);
        if (num_ver == 0) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("No vertices found in the mesh");
        }

        // Coordinates array
        py::array_t<double> coords(std::vector<py::ssize_t>{num_ver, dim});
        auto coords_ptr = coords.mutable_data();

        // File position of vertices
        if (!GmfGotoKwd(mesh_id, GmfVertices)) {
            GmfCloseMesh(mesh_id);
            throw std::runtime_error("Failed to find vertices in the mesh");
        }

        // Read vertices
        for (int64_t i = 0; i < num_ver; i++) {
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
            {GmfEdges, 2},
            {GmfTriangles, 3},
            {GmfQuadrilaterals, 4},
            {GmfTetrahedra, 4},
            {GmfPrisms, 6},
            {GmfHexahedra, 8}
        };

        for (const auto& [kwd, num_nodes] : element_types) {
            int64_t num_elm = GmfStatKwd(mesh_id, kwd);
            if (num_elm > 0) {
                py::array_t<int64_t> element_array(std::vector<py::ssize_t>{num_elm, num_nodes + 1});
                auto elm_ptr = element_array.mutable_data();

                if (GmfGotoKwd(mesh_id, kwd)) {
                    for (int64_t i = 0; i < num_elm; i++) {
                        std::vector<int> bufInt(num_nodes);
                        int ref;
                        if (kwd == GmfEdges) {
                            GmfGetLin(mesh_id, GmfEdges,
                                      &bufInt[0], &bufInt[1], &ref);
                        } else if (kwd == GmfTriangles) {
                            GmfGetLin(mesh_id, GmfTriangles,
                                      &bufInt[0], &bufInt[1], &bufInt[2], &ref);
                        } else if (kwd == GmfQuadrilaterals) {
                            GmfGetLin(mesh_id, GmfQuadrilaterals,
                                      &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &ref);
                        } else if (kwd == GmfTetrahedra) {
                            GmfGetLin(mesh_id, GmfTetrahedra,
                                      &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3], &ref);
                        } else if (kwd == GmfPrisms) {
                            GmfGetLin(mesh_id, GmfPrisms,
                                      &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3],
                                      &bufInt[4], &bufInt[5], &ref);
                        } else if (kwd == GmfHexahedra) {
                            GmfGetLin(mesh_id, GmfHexahedra,
                                      &bufInt[0], &bufInt[1], &bufInt[2], &bufInt[3],
                                      &bufInt[4], &bufInt[5], &bufInt[6], &bufInt[7], &ref);
                        }

                        for (int j = 0; j < num_nodes; j++) {
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
                else throw std::runtime_error("Unknown keyword");

                elements[key.c_str()] = element_array;
            }
        }

        // Close the mesh
        GmfCloseMesh(mesh_id);

        // Read solution if requested
        py::dict sol;

        if (read_sol) {
            // Open mesh for reading
            int64_t sol_id = GmfOpenMesh(solpath.c_str(), GmfRead, &ver, &dim);
            if (sol_id == 0) {
                throw std::runtime_error("Failed to open solution: " + solpath);
            }

            int num_types, sol_size;
            int types[GmfMaxTyp];

            // Check if solution exist
            int64_t num_lin = GmfStatKwd(mesh_id, GmfSolAtVertices, &num_types, &sol_size, types);

            if (num_lin > 0) {
                // Solution file position
                if (GmfGotoKwd(mesh_id, GmfSolAtVertices)) {
                    // Solution array
                    std::vector<py::array_t<double>> sol_array;
                    py::array_t<double> bufDbl(sol_size);
                    for (int64_t i = 0; i < num_ver; i++) {
                        GmfGetLin(mesh_id, GmfSolAtVertices, &bufDbl);
                        sol_array.push_back(bufDbl);
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
                            py::array_t<double> vector_field(std::vector<py::ssize_t>{num_ver, dim});
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
                            py::array_t<double> matrix_field(std::vector<py::ssize_t>{num_ver, sym_size});
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

        // Return coords and solution if available
        if (read_sol && sol.size() > 0) {
            return py::make_tuple(coords, elements, sol);
        } else {
            return py::make_tuple(coords, elements);
        }
    }, py::arg("meshpath"), py::arg("solpath") = "", py::arg("read_sol") = false,
       "Read a meshb file and return nodes and coordinates as numpy arrays");

       m.def("write_mesh", [](const std::string& meshpath, py::array_t<double> coords,
                              py::dict elements, const std::string& solpath = "",
                              py::dict sol = py::dict()) {
        int ver = 4;
        int dim = coords.shape(1);

        // Create output directory if needed
        size_t last_slash = meshpath.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string dir = meshpath.substr(0, last_slash);
            std::string cmd = "mkdir -p " + dir;
            int ret = system(cmd.c_str());
            if (ret != 0) {
                throw std::runtime_error("Failed to create directory: " + dir);
            }
        }

        // Open mesh for writing
        int64_t mesh_id = GmfOpenMesh(meshpath.c_str(), GmfWrite, ver, dim);
        if (mesh_id == 0) {
            throw std::runtime_error("Failed to open mesh for writing: " + meshpath);
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
                GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim],
                          coords_ptr[i * dim + 1], 0);
            } else {
                GmfSetLin(mesh_id, GmfVertices, coords_ptr[i * dim],
                          coords_ptr[i * dim + 1], coords_ptr[i * dim + 2], 0);
            }
        }

        // Write elements
        for (auto item : elements) {
            std::string key = item.first.cast<std::string>();
            py::array_t<int64_t> element_array = item.second.cast<py::array_t<int64_t>>();
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

            for (int64_t i = 0; i < num_elm; i++) {
                int j = i * (num_nodes + 1);
                if (kwd == GmfEdges) {
                    GmfSetLin(mesh_id, GmfEdges,
                            elm_ptr[j    ], elm_ptr[j + 1], elm_ptr[j + 2]);
                } else if (kwd == GmfTriangles) {
                    GmfSetLin(mesh_id, GmfTriangles,
                            elm_ptr[j    ], elm_ptr[j + 1], elm_ptr[j + 2], elm_ptr[j + 3]);
                } else if (kwd == GmfQuadrilaterals) {
                    GmfSetLin(mesh_id, GmfQuadrilaterals,
                            elm_ptr[j    ], elm_ptr[j + 1], elm_ptr[j + 2], elm_ptr[j + 3], elm_ptr[j + 4]);
                } else if (kwd == GmfTetrahedra) {
                    GmfSetLin(mesh_id, GmfTetrahedra,
                            elm_ptr[j    ], elm_ptr[j + 1], elm_ptr[j + 2], elm_ptr[j + 3], elm_ptr[j + 4]);
                } else if (kwd == GmfPrisms) {
                    GmfSetLin(mesh_id, GmfPrisms,
                            elm_ptr[j], elm_ptr[j + 1], elm_ptr[j + 2], elm_ptr[j + 3],
                            elm_ptr[j + 4], elm_ptr[j + 5], elm_ptr[j + 6]);
                } else if (kwd == GmfHexahedra) {
                    GmfSetLin(mesh_id, GmfHexahedra,
                            elm_ptr[j    ], elm_ptr[j + 1], elm_ptr[j + 2], elm_ptr[j + 3],
                            elm_ptr[j + 4], elm_ptr[j + 5], elm_ptr[j + 6], elm_ptr[j + 7], elm_ptr[j + 8]);
                }
            }
        }

        // Close the mesh
        GmfCloseMesh(mesh_id);

        // Write solution if provided
        if (!sol.empty()) {
            // Open solution for writing
            int64_t sol_id = GmfOpenMesh(solpath.c_str(), GmfWrite, ver, dim);
            if (sol_id == 0) {
                throw std::runtime_error("Failed to open solution: " + solpath);
            }

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
                GmfSetKwd(sol_id, GmfSolAtVertices, num_ver, field_type.size(), field_type.data());

                // Write solution values for each vertex
                std::vector<double> bufDbl;
                for (int64_t i = 0; i < num_ver; i++) {
                    bufDbl.clear();

                    for (size_t j = 0; j < field_type.size(); j++) {
                        auto field_ptr = field_array[j].data();

                        if (field_type[j] == GmfSca) {
                            // Add scalar value to buffer
                            bufDbl.push_back(field_ptr[i]);
                        } else if (field_type[j] == GmfVec) {
                            // Add vector values to buffer
                            for (int k = 0; k < dim; k++) {
                                bufDbl.push_back(field_ptr[i * dim + k]);
                            }
                        } else if (field_type[j] == GmfSymMat) {
                            // Add symmetric matrix values to buffer
                            int sym_size = (dim * (dim + 1)) / 2;
                            for (int k = 0; k < sym_size; k++) {
                                bufDbl.push_back(field_ptr[i * sym_size + k]);
                            }
                        }
                    }

                    // Write the buffer for the current vertex
                    GmfSetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                }
            }

            // Close the solution
            GmfCloseMesh(sol_id);
        }

        return true;
    }, py::arg("meshpath"), py::arg("coords"), py::arg("elements"),
       py::arg("solpath") = "", py::arg("sol") = py::dict(),
       "Write nodes and coordinates to a meshb file");
}