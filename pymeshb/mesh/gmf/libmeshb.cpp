/** @file libmeshb.cpp
 *  @brief Python bindings for libMeshb library.
 *
 *  Implementation of Python bindings for the libMeshb library, providing
 *  functionality to read and write meshb files, including scalar, vector,
 *  and tensor fields.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
extern "C" {
#include <libmeshb7.h>
}

namespace py = pybind11;

/**
 * Python module definition for libMeshb bindings.
 */
PYBIND11_MODULE(libmeshb, m) {
    m.doc() = "Python bindings for libMeshb";

    /**
     * Read mesh data from a meshb file.
     *
     * @param meshpath Path to the mesh file
     * @param solpath Path to the solution file (optional)
     * @param read_sol Whether to read solution data (default: false)
     * @return Tuple containing coordinates, elements, and optionally solution data
     */
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
                py::array_t<unsigned int> element_array(std::vector<py::ssize_t>{num_elm, num_nodes + 1});
                auto elm_ptr = element_array.mutable_data();

                if (GmfGotoKwd(mesh_id, kwd)) {
                    for (auto i = 0; i < num_elm; i++) {
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
            int64_t num_lin = GmfStatKwd(sol_id, GmfSolAtVertices, &num_types, &sol_size, types);

            if (num_lin > 0) {
                // Solution field name file position
                int num_sol_name = GmfStatKwd(sol_id, GmfReferenceStrings);
                std::vector<std::string> sol_name;
                std::vector<int> sol_ind;
                if (num_sol_name > 0 and GmfGotoKwd(sol_id, GmfReferenceStrings)) {
                    int kwd, ind;
                    char bufChar[4096];
                    for (auto i = 0; i < num_sol_name; i++) {
                        GmfGetLin(sol_id, GmfReferenceStrings, &kwd, &ind, &bufChar);
                        // Convert to string and remove spaces/newlines
                        std::string field_name = std::string(bufChar, strnlen(bufChar, sizeof(bufChar)));
                        field_name.erase(std::remove_if(field_name.begin(),
                                         field_name.end(),
                                         [](char c){ return std::isspace(c); }),
                                         field_name.end());
                        sol_name.push_back(field_name);
                        sol_ind.push_back(ind-1);
                    }
                }

                // Solution file position
                if (GmfGotoKwd(sol_id, GmfSolAtVertices)) {
                    std::vector<double> bufDbl(sol_size);
                    int field_start = 0;
                    for (auto i = 0; i < num_types; i++) {
                        // Determine field name
                        std::string field_name = "sol_" + std::to_string(i);
                        auto it = std::find(sol_ind.begin(), sol_ind.end(), i);
                        if (it != sol_ind.end()) {
                            int index = std::distance(sol_ind.begin(), it);
                            field_name = sol_name[index];
                        }

                        // Create appropriate array based on field type
                        if (types[i] == GmfSca) {
                            // Scalar field - 1D array
                            py::array_t<double> scalar_field(num_ver);
                            auto scalar_ptr = scalar_field.mutable_data();

                            // Reset file position to start of solution data
                            GmfGotoKwd(sol_id, GmfSolAtVertices);

                            // Read scalar values for each vertex
                            for (auto j = 0; j < num_ver; j++) {
                                GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                                scalar_ptr[j] = bufDbl[field_start];
                            }

                            sol[field_name.c_str()] = scalar_field;
                            field_start += 1;

                        } else if (types[i] == GmfVec) {
                            // Vector field - 2D array (num_ver x dim)
                            py::array_t<double> vector_field(std::vector<py::ssize_t>{num_ver, dim});
                            auto vector_ptr = vector_field.mutable_data();

                            // Reset file position to start of solution data
                            GmfGotoKwd(sol_id, GmfSolAtVertices);

                            // Read vector values for each vertex
                            for (auto j = 0; j < num_ver; j++) {
                                GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                                for (auto k = 0; k < dim; k++) {
                                    vector_ptr[j*dim + k] = bufDbl[field_start + k];
                                }
                            }

                            sol[field_name.c_str()] = vector_field;
                            field_start += dim;

                        } else if (types[i] == GmfSymMat) {
                            // Symmetric matrix field
                            int sym_size = (dim*(dim+1))/2;
                            py::array_t<double> matrix_field(std::vector<py::ssize_t>{num_ver, sym_size});
                            auto matrix_ptr = matrix_field.mutable_data();

                            // Reset file position to start of solution data
                            GmfGotoKwd(sol_id, GmfSolAtVertices);

                            // Read symmetric matrix values for each vertex
                            for (auto j = 0; j < num_ver; j++) {
                                GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                                for (auto k = 0; k < sym_size; k++) {
                                    matrix_ptr[j*sym_size + k] = bufDbl[field_start + k];
                                }
                            }

                            sol[field_name.c_str()] = matrix_field;
                            field_start += sym_size;
                        }
                    }
                }
            } // num_lin > 0
            else {
                GmfCloseMesh(sol_id);
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

       /**
         * Write mesh data to a meshb file.
         *
         * @param meshpath Path to the mesh file
         * @param coords Coordinates of each node
         * @param elements Dictionary of mesh elements
         * @param solpath Path to the solution file (optional)
         * @param sol Dictionary of solution data (optional)
         * @return Boolean indicating success
         */
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

            std::vector<int> sol_type;
            std::vector<std::string> sol_name;
            std::vector<py::array_t<double>> sol_array;

            for (auto item : sol) {
                auto key = item.first.cast<std::string>();
                auto field = item.second.cast<py::array_t<double>>();

                // Determine field type from array shape
                if (field.ndim() == 1) {
                    // Scalar field
                    sol_type.push_back(GmfSca);
                } else if (field.ndim() == 2 && field.shape(1) == dim) {
                    // Vector field
                    sol_type.push_back(GmfVec);
                } else if (field.ndim() == 2 && field.shape(1) == (dim * (dim + 1)) / 2) {
                    // Symmetric matrix field
                    sol_type.push_back(GmfSymMat);
                } else {
                    continue;
                }

                sol_name.push_back(key);
                sol_array.push_back(field);
            }

            // Write solution if any valid fields were found
            if (!sol_type.empty()) {
                // Set solution field names
                GmfSetKwd(sol_id, GmfReferenceStrings, sol_name.size());
                for (auto i = 0; i < sol_name.size(); i++) {
                    GmfSetLin(sol_id, GmfReferenceStrings, GmfSolAtVertices, i + 1, sol_name[i].c_str());
                }

                // Set solution keyword with field types
                GmfSetKwd(sol_id, GmfSolAtVertices, num_ver, sol_type.size(), sol_type.data());

                // Write solution values for each vertex
                std::vector<double> bufDbl;
                for (auto i = 0; i < num_ver; i++) {
                    bufDbl.clear();

                    for (size_t j = 0; j < sol_type.size(); j++) {
                        auto sol_ptr = sol_array[j].data();

                        if (sol_type[j] == GmfSca) {
                            // Add scalar value to buffer
                            bufDbl.push_back(sol_ptr[i]);
                        } else if (sol_type[j] == GmfVec) {
                            // Add vector values to buffer
                            for (auto k = 0; k < dim; k++) {
                                bufDbl.push_back(sol_ptr[i * dim + k]);
                            }
                        } else if (sol_type[j] == GmfSymMat) {
                            // Add symmetric matrix values to buffer
                            int sym_size = (dim * (dim + 1)) / 2;
                            for (auto k = 0; k < sym_size; k++) {
                                bufDbl.push_back(sol_ptr[i * sym_size + k]);
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