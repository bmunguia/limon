#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ref_map.hpp"
#include "solution.hpp"
#include "../util.hpp"

namespace pymeshb {
namespace su2 {

// Constants for SU2 binary file format
const int SU2_MAGIC_NUMBER = 535532;  // Hex representation of "SU2"
const int CGNS_STRING_SIZE = 33;      // Fixed size for variable names in SU2 files

py::dict read_solution_ascii(const std::string& solpath, int num_point, int dim) {
    py::dict sol;

    std::ifstream sol_file(solpath);
    if (!sol_file.is_open()) {
        return sol;
    }

    // Parse CSV header for field names
    std::string header;
    std::getline(sol_file, header);

    // Split and clean header fields
    std::vector<std::string> field_names;
    std::stringstream ss(header);
    std::string field;
    while (std::getline(ss, field, ',')) {
        field = pymeshb::trim(field);
        // Remove quotes if present
        if (field.front() == '"' && field.back() == '"') {
            field = field.substr(1, field.size() - 2);
        }
        field_names.push_back(field);
    }

    // Read data into temporary storage
    std::vector<std::vector<double>> data(field_names.size());
    std::string line;
    while (std::getline(sol_file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',') && col < field_names.size()) {
            data[col].push_back(std::stod(pymeshb::trim(cell)));
            col++;
        }
    }

    // Process fields
    sol = process_solution_fields(field_names, data, num_point, dim);

    return sol;
}

py::dict read_solution_binary(const std::string& solpath, int num_point, int dim) {
    py::dict sol;

    std::ifstream sol_file(solpath, std::ios::in | std::ios::binary);
    if (!sol_file.is_open()) {
        return sol;
    }

    // Read the magic number to verify it's an SU2 binary file
    int magic_number = 0;
    sol_file.read(reinterpret_cast<char*>(&magic_number), sizeof(int));

    // Check that this is an SU2 binary file
    if (magic_number != SU2_MAGIC_NUMBER) {
        std::cerr << "Error: File " << solpath << " is not a valid SU2 binary restart file." << std::endl;
        return sol;
    }

    // Read number of variables
    int num_var = 0;
    sol_file.read(reinterpret_cast<char*>(&num_var), sizeof(int));

    // Read number of points (skip, as we already know it's num_point)
    int num_point_sol = 0;
    sol_file.read(reinterpret_cast<char*>(&num_point_sol), sizeof(int));

    // Skip two extra integers that might be present in newer SU2 versions
    int dummy;
    sol_file.read(reinterpret_cast<char*>(&dummy), sizeof(int));
    sol_file.read(reinterpret_cast<char*>(&dummy), sizeof(int));

    // Read variable names
    std::vector<std::string> field_names;
    field_names.reserve(num_var);

    for (int ivar = 0; ivar < num_var; ivar++) {
        // Allocate a fixed-size buffer for the variable name
        char str_buf[CGNS_STRING_SIZE];
        memset(str_buf, 0, CGNS_STRING_SIZE);

        // Read the entire fixed-size block for the variable name
        sol_file.read(str_buf, CGNS_STRING_SIZE);

        // Create a string from the buffer (will stop at first null character)
        std::string field_name(str_buf);

        // Validate the name (ensure it's not empty and contains only printable characters)
        bool valid = !field_name.empty();
        for (char c : field_name) {
            if (!std::isprint(static_cast<unsigned char>(c))) {
                valid = false;
                break;
            }
        }

        if (valid) {
            field_names.push_back(field_name);
        } else {
            // Use a default name when invalid
            field_names.push_back("Variable_" + std::to_string(ivar));
        }
    }

    // Read data
    std::vector<std::vector<double>> data(field_names.size());
    for (auto& column : data) {
        column.resize(num_point, 0.0);
    }
    for (int ipoint = 0; ipoint < num_point; ipoint++) {
        for (int ivar = 0; ivar < num_var; ivar++) {
            double value;
            sol_file.read(reinterpret_cast<char*>(&value), sizeof(double));
            data[ivar][ipoint] = value;
        }
    }

    // Process fields
    sol = process_solution_fields(field_names, data, num_point, dim);

    return sol;
}

py::dict process_solution_fields(const std::vector<std::string>& field_names,
                                const std::vector<std::vector<double>>& data,
                                int num_point, int dim) {
    int sym_size = (dim * (dim + 1)) / 2;
    std::vector<std::string> vec_suffixes = {"_x", "_y", "_z"};
    std::vector<std::string> mat_suffixes = {"_xx", "_xy", "_yy", "_xz", "_yz", "_zz"};

    py::dict sol;

    size_t col_index = 0;

    // Skip the PointID column if present
    if (col_index < field_names.size() &&
        (field_names[col_index] == "PointID" || field_names[col_index] == "\"PointID\"")) {
        col_index++;
    }

    while (col_index < field_names.size()) {
        std::string field_name = field_names[col_index];

        // Strip quotes if present
        if (field_name.front() == '"' && field_name.back() == '"') {
            field_name = field_name.substr(1, field_name.size() - 2);
        }

        // Check if it's a vector field
        if (col_index + dim <= field_names.size()) {
            std::string base_name = field_name;
            if (base_name.length() > 2 && base_name.substr(base_name.length() - 2) == vec_suffixes[0]) {
                base_name = base_name.substr(0, base_name.length() - 2);

                bool is_vector = true;
                for (size_t j = 0; j < dim; j++) {
                    if (col_index + j >= field_names.size() ||
                        field_names[col_index + j] != base_name + vec_suffixes[j]) {
                        is_vector = false;
                        break;
                    }
                }

                if (is_vector) {
                    // Create vector field
                    py::array_t<double> vector_field(std::vector<py::ssize_t>{num_point, dim});
                    auto vector_ptr = vector_field.mutable_data();

                    // Copy data
                    for (int i = 0; i < num_point && i < data[col_index].size(); i++) {
                        for (int j = 0; j < dim; j++) {
                            vector_ptr[i * dim + j] = data[col_index + j][i];
                        }
                    }

                    sol[base_name.c_str()] = vector_field;
                    col_index += dim;
                    continue;
                }
            }
        }

        // Check if it's a symmetric matrix field
        if (col_index + sym_size <= field_names.size()) {
            std::string base_name = field_name;
            if (base_name.length() > 3 && base_name.substr(base_name.length() - 3) == mat_suffixes[0]) {
                base_name = base_name.substr(0, base_name.length() - 3);

                bool is_matrix = true;
                for (size_t j = 0; j < sym_size; j++) {
                    if (col_index + j >= field_names.size() ||
                        field_names[col_index + j] != base_name + mat_suffixes[j]) {
                        is_matrix = false;
                        break;
                    }
                }

                if (is_matrix) {
                    // Create matrix field
                    py::array_t<double> matrix_field(std::vector<py::ssize_t>{num_point, sym_size});
                    auto matrix_ptr = matrix_field.mutable_data();

                    // Copy data
                    for (int i = 0; i < num_point && i < data[col_index].size(); i++) {
                        for (int j = 0; j < sym_size; j++) {
                            matrix_ptr[i * sym_size + j] = data[col_index + j][i];
                        }
                    }

                    sol[base_name.c_str()] = matrix_field;
                    col_index += sym_size;
                    continue;
                }
            }
        }

        // Default: treat as scalar field
        py::array_t<double> scalar_field(num_point);
        auto scalar_ptr = scalar_field.mutable_data();

        for (int i = 0; i < num_point && i < data[col_index].size(); i++) {
            scalar_ptr[i] = data[col_index][i];
        }

        sol[field_name.c_str()] = scalar_field;
        col_index++;
    }

    return sol;
}

py::dict read_solution(const std::string& solpath, const std::string& labelpath, int num_point, int dim) {
    py::dict sol;

    std::ifstream file_check(solpath, std::ios::in | std::ios::binary);
    if (!file_check.is_open()) {
        return sol;
    }

    // Check for the SU2 magic number
    int magic_number = 0;
    file_check.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    file_check.close();

    if (magic_number == SU2_MAGIC_NUMBER) {
        sol = read_solution_binary(solpath, num_point, dim);
    } else {
        sol = read_solution_ascii(solpath, num_point, dim);
    }

    // Save updated marker map back to file if provided
    if (!labelpath.empty()) {
        std::map<int, std::string> ref_map;
        int ref_id = 1;
        for (auto& [sol_id, _] : sol) {
            ref_map[ref_id] = sol_id.cast<std::string>();
            ref_id++;
        }
        RefMap::saveRefMap(ref_map, labelpath);
    }

    return sol;
}

bool write_solution_ascii(const std::string& solpath, py::dict sol, int num_point, int dim) {
    int sym_size = (dim * (dim + 1)) / 2;
    std::vector<std::string> vec_suffixes = {"_x", "_y", "_z"};
    std::vector<std::string> mat_suffixes = {"_xx", "_xy", "_yy", "_xz", "_yz", "_zz"};

    if (!pymeshb::createDirectory(solpath)) {
        return false;
    }

    std::ofstream sol_file(solpath);
    if (!sol_file.is_open()) {
        return false;
    }
    sol_file.precision(15);

    // Prepare data vectors and column headers
    std::vector<std::string> headers;
    std::vector<std::vector<double>> data_columns;

    // Add PointID columnn
    headers.push_back("\"PointID\"");
    std::vector<double> point_ids(num_point);
    for (int i = 0; i < num_point; i++) {
        point_ids[i] = static_cast<double>(i);
    }
    data_columns.push_back(point_ids);

    // Process each field
    for (auto item : sol) {
        std::string key = item.first.cast<std::string>();
        py::array_t<double> field = item.second.cast<py::array_t<double>>();

        if (field.ndim() == 1) {
            // Scalar field
            headers.push_back("\"" + key + "\"");
            std::vector<double> column(num_point);
            auto field_ptr = field.data();
            for (int i = 0; i < num_point; i++) {
                column[i] = field_ptr[i];
            }
            data_columns.push_back(column);
        }
        else if (field.ndim() == 2 && field.shape(1) == dim) {
            // Vector field
            auto field_ptr = field.data();
            for (int j = 0; j < dim; j++) {
                headers.push_back("\"" + key + vec_suffixes[j] + "\"");
                std::vector<double> column(num_point);
                for (int i = 0; i < num_point; i++) {
                    column[i] = field_ptr[i * dim + j];
                }
                data_columns.push_back(column);
            }
        }
        else if (field.ndim() == 2 && field.shape(1) == sym_size) {
            // Symmetric matrix field
            auto field_ptr = field.data();
            for (int j = 0; j < sym_size; j++) {
                headers.push_back("\"" + key + mat_suffixes[j] + "\"");
                std::vector<double> column(num_point);
                for (int i = 0; i < num_point; i++) {
                    column[i] = field_ptr[i * sym_size + j];
                }
                data_columns.push_back(column);
            }
        }
    }

    // Write headers
    if (!headers.empty()) {
        sol_file << headers[0];
        for (size_t i = 1; i < headers.size(); i++) {
            sol_file << "," << headers[i];
        }
        sol_file << std::endl;

        // Write data rows
        for (int i = 0; i < num_point; i++) {
            sol_file << static_cast<int>(data_columns[0][i]);
            for (size_t j = 1; j < data_columns.size(); j++) {
                sol_file << ", " << std::scientific << data_columns[j][i];
            }
            sol_file << std::endl;
        }
    }

    sol_file.close();
    return true;
}

bool write_solution_binary(const std::string& solpath, py::dict sol, int num_point, int dim) {
    int sym_size = (dim * (dim + 1)) / 2;
    std::vector<std::string> vec_suffixes = {"_x", "_y", "_z"};
    std::vector<std::string> mat_suffixes = {"_xx", "_xy", "_yy", "_xz", "_yz", "_zz"};

    if (!pymeshb::createDirectory(solpath)) {
        return false;
    }

    std::ofstream sol_file(solpath, std::ios::out | std::ios::binary);
    if (!sol_file.is_open()) {
        return false;
    }

    // Prepare variable names and data
    std::vector<std::string> var_names;
    std::vector<std::vector<double>> data_columns;

    // Process each field
    for (auto item : sol) {
        std::string key = item.first.cast<std::string>();
        py::array_t<double> field = item.second.cast<py::array_t<double>>();

        if (field.ndim() == 1) {
            // Scalar field
            var_names.push_back(key);
            std::vector<double> column(num_point);
            auto field_ptr = field.data();
            for (int i = 0; i < num_point; i++) {
                column[i] = field_ptr[i];
            }
            data_columns.push_back(column);
        }
        else if (field.ndim() == 2 && field.shape(1) == dim) {
            // Vector field
            auto field_ptr = field.data();
            for (int j = 0; j < dim; j++) {
                var_names.push_back(key + vec_suffixes[j]);
                std::vector<double> column(num_point);
                for (int i = 0; i < num_point; i++) {
                    column[i] = field_ptr[i * dim + j];
                }
                data_columns.push_back(column);
            }
        }
        else if (field.ndim() == 2 && field.shape(1) == sym_size) {
            // Symmetric matrix field
            auto field_ptr = field.data();
            for (int j = 0; j < sym_size; j++) {
                var_names.push_back(key + mat_suffixes[j]);
                std::vector<double> column(num_point);
                for (int i = 0; i < num_point; i++) {
                    column[i] = field_ptr[i * sym_size + j];
                }
                data_columns.push_back(column);
            }
        }
    }

    // Number of variables
    int num_var = static_cast<int>(var_names.size());

    // Write header with magic number and metadata
    // Buffer contains magic number, num_var, num_point, and two zeros
    int var_buf[5] = {SU2_MAGIC_NUMBER, num_var, num_point, 0, 0};
    sol_file.write(reinterpret_cast<const char*>(var_buf), 5 * sizeof(int));

    // Variable names
    for (const auto& name : var_names) {
        // Create a fixed-size buffer for the name
        char str_buf[CGNS_STRING_SIZE];
        memset(str_buf, 0, CGNS_STRING_SIZE);

        // Copy the name to the buffer (limited to CGNS_STRING_SIZE-1 to ensure null termination)
        strncpy(str_buf, name.c_str(), CGNS_STRING_SIZE-1);

        // Write the entire fixed-size block
        sol_file.write(str_buf, CGNS_STRING_SIZE);
    }

    // Solution data
    for (int ipoint = 0; ipoint < num_point; ipoint++) {
        for (int ivar = 0; ivar < num_var; ivar++) {
            double value = data_columns[ivar][ipoint];
            sol_file.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }

    sol_file.close();
    return true;
}

bool write_solution(const std::string& solpath, const std::string& labelpath, py::dict sol, int num_point, int dim) {
    // Check file extension to determine format
    std::string extension;
    size_t dot_pos = solpath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        extension = solpath.substr(dot_pos);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                      [](unsigned char c){ return std::tolower(c); });
    }

    if (extension == ".dat") {
        return write_solution_binary(solpath, sol, num_point, dim);
    } else {
        // Default to ASCII
        return write_solution_ascii(solpath, sol, num_point, dim);
    }
}

}  // namespace su2
}  // namespace pymeshb