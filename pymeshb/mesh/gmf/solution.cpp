#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libmeshb7.h>
}

#include "solution.hpp"
#include "../ref_map.hpp"
#include "../util.hpp"

namespace pymeshb {
namespace gmf {

py::dict read_solution(const std::string& solpath, int64_t num_ver, int dim, 
                       const std::string& labelpath) {
    py::dict sol;
    int version;
    int dim_sol;

    // Load solution label map from file if provided
    std::map<int, std::string> ref_map;
    if (!labelpath.empty()) {
        ref_map = RefMap::loadRefMap(labelpath);
    }

    // Open the solution
    int64_t sol_id = GmfOpenMesh(solpath.c_str(), GmfRead, &version, &dim_sol);
    if (sol_id == 0) {
        throw std::runtime_error("Failed to open solution: " + solpath);
    }

    int num_types, sol_size_total;
    int types[GmfMaxTyp];

    int sym_size = (dim * (dim + 1)) / 2;

    int64_t num_lin = GmfStatKwd(sol_id, GmfSolAtVertices, &num_types, &sol_size_total, types);

    if (num_lin > 0) {
        // Solution file position
        if (GmfGotoKwd(sol_id, GmfSolAtVertices)) {
            std::vector<double> bufDbl(sol_size_total);
            int field_start_offset = 0;
            for (auto i = 0; i < num_types; i++) {
                // Load label name or fallback to REF_<marker_id>
                std::string field_name = RefMap::getRefName(ref_map, i + 1);

                if (types[i] == GmfSca) {
                    py::array_t<double> scalar_field(num_ver);
                    auto scalar_ptr = scalar_field.mutable_data();
                    GmfGotoKwd(sol_id, GmfSolAtVertices);
                    for (auto j = 0; j < num_ver; j++) {
                        GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                        scalar_ptr[j] = bufDbl[field_start_offset];
                    }
                    sol[field_name.c_str()] = scalar_field;
                    field_start_offset += 1;
                } else if (types[i] == GmfVec) {
                    py::array_t<double> vector_field(std::vector<py::ssize_t>{num_ver, (long)dim});
                    auto vector_ptr = vector_field.mutable_data();
                    GmfGotoKwd(sol_id, GmfSolAtVertices);
                    for (auto j = 0; j < num_ver; j++) {
                        GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                        for (auto k = 0; k < dim; k++) {
                            vector_ptr[j * dim + k] = bufDbl[field_start_offset + k];
                        }
                    }
                    sol[field_name.c_str()] = vector_field;
                    field_start_offset += dim;
                } else if (types[i] == GmfSymMat) {
                    py::array_t<double> matrix_field(std::vector<py::ssize_t>{num_ver, (long)sym_size});
                    auto matrix_ptr = matrix_field.mutable_data();
                    GmfGotoKwd(sol_id, GmfSolAtVertices);
                    for (auto j = 0; j < num_ver; j++) {
                        GmfGetLin(sol_id, GmfSolAtVertices, bufDbl.data());
                        for (auto k = 0; k < sym_size; k++) {
                            matrix_ptr[j * sym_size + k] = bufDbl[field_start_offset + k];
                        }
                    }
                    sol[field_name.c_str()] = matrix_field;
                    field_start_offset += sym_size;
                }
            }
        }
    }

    // Close the solution
    GmfCloseMesh(sol_id);

    return sol;
}

bool write_solution(const std::string& solpath, py::dict sol_data, int64_t num_ver, 
                    int dim, const std::string& labelpath) {
    if (sol_data.empty()) {
        return true;
    }

    // Open the solution
    int version = 2;
    int64_t sol_id = GmfOpenMesh(solpath.c_str(), GmfWrite, version, dim);
    if (sol_id == 0) {
        throw std::runtime_error("Failed to open solution for writing: " + solpath);
    }

    std::vector<int> sol_types;
    std::vector<py::array_t<double>> sol_field_arrays;
    std::map<int, std::string> ref_map;

    int sym_size = (dim * (dim + 1)) / 2;

    int ref_id = 1;
    for (auto item : sol_data) {
        auto key = item.first.cast<std::string>();
        auto field_array = item.second.cast<py::array_t<double>>();

        // Determine field type from array shape
        if (field_array.ndim() == 1) {
            // Scalar field
            sol_types.push_back(GmfSca);
        } else if (field_array.ndim() == 2 && field_array.shape(1) == dim) {
            // Vector field
            sol_types.push_back(GmfVec);
        } else if (field_array.ndim() == 2 && field_array.shape(1) == sym_size) {
            // Symmetric matrix field
            sol_types.push_back(GmfSymMat);
        } else {
            GmfCloseMesh(sol_id);
            throw std::runtime_error("Unsupported solution field shape for: " + key);
        }
        ref_map[ref_id++] = key;
        sol_field_arrays.push_back(field_array);
    }

    // Write solution if any valid fields were found
    if (!sol_types.empty()) {
        // Set solution keyword with field types
        GmfSetKwd(sol_id, GmfSolAtVertices, num_ver, sol_types.size(), sol_types.data());

        // Write solution values for each vertex
        std::vector<double> bufDbl;
        for (auto i = 0; i < num_ver; i++) {
            bufDbl.clear();
            for (size_t j = 0; j < sol_types.size(); j++) {
                // Add scalar value to buffer
                auto field_ptr = sol_field_arrays[j].data();
                if (sol_types[j] == GmfSca) {
                    bufDbl.push_back(field_ptr[i]);
                } else if (sol_types[j] == GmfVec) {
                    // Add vector values to buffer
                    for (auto k = 0; k < dim; k++) {
                        bufDbl.push_back(field_ptr[i * dim + k]);
                    }
                } else if (sol_types[j] == GmfSymMat) {
                    // Add symmetric matrix values to buffer
                    for (auto k = 0; k < sym_size; k++) {
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

    // Save updated solution label map back to file if provided
    if (!labelpath.empty()) {
        RefMap::saveRefMap(ref_map, labelpath);
    }

    return true;
}

} // namespace gmf
} // namespace pymeshb