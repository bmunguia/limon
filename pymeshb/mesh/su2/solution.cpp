#include "solution.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

#include "../util.hpp"

namespace pymeshb {
namespace su2 {

py::dict read_solution(const std::string& solpath, int num_ver, int dim) {
    py::dict sol;

    std::ifstream sol_file(solpath);
    if (!sol_file.is_open()) {
        return sol;  // Return empty dictionary if file can't be opened
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

    // Check for vector fields (X,Y,Z components)
    size_t col_index = 0;
    while (col_index < field_names.size()) {
        std::string field_name = field_names[col_index];

        // Check if it's a vector field
        if (col_index + dim <= field_names.size()) {
            std::string base_name = field_name;
            if (base_name.length() > 2 && base_name.substr(base_name.length() - 2) == "_X") {
                base_name = base_name.substr(0, base_name.length() - 2);

                if ((col_index + 1 < field_names.size() &&
                     field_names[col_index + 1] == base_name + "_Y") &&
                    (dim == 2 || (col_index + 2 < field_names.size() &&
                                  field_names[col_index + 2] == base_name + "_Z"))) {

                    // Create vector field
                    py::array_t<double> vector_field(std::vector<py::ssize_t>{num_ver, dim});
                    auto vector_ptr = vector_field.mutable_data();

                    // Copy data
                    for (int i = 0; i < num_ver && i < data[col_index].size(); i++) {
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

        // Check if it's a symmetric tensor field
        if (col_index + (dim*(dim+1))/2 <= field_names.size()) {
            std::string base_name = field_name;
            if (base_name.length() > 3 && base_name.substr(base_name.length() - 3) == "_XX") {
                base_name = base_name.substr(0, base_name.length() - 3);

                bool is_tensor = true;
                std::vector<std::string> tensor_suffixes;

                if (dim == 2) {
                    tensor_suffixes = {"_XX", "_XY", "_YY"};
                } else {
                    tensor_suffixes = {"_XX", "_XY", "_XZ", "_YY", "_YZ", "_ZZ"};
                }

                for (size_t j = 0; j < tensor_suffixes.size(); j++) {
                    if (col_index + j >= field_names.size() ||
                        field_names[col_index + j] != base_name + tensor_suffixes[j]) {
                        is_tensor = false;
                        break;
                    }
                }

                if (is_tensor) {
                    // Create tensor field
                    int sym_size = (dim*(dim+1))/2;
                    py::array_t<double> tensor_field(std::vector<py::ssize_t>{num_ver, sym_size});
                    auto tensor_ptr = tensor_field.mutable_data();

                    // Copy data
                    for (int i = 0; i < num_ver && i < data[col_index].size(); i++) {
                        for (int j = 0; j < sym_size; j++) {
                            tensor_ptr[i * sym_size + j] = data[col_index + j][i];
                        }
                    }

                    sol[base_name.c_str()] = tensor_field;
                    col_index += sym_size;
                    continue;
                }
            }
        }

        // Default: treat as scalar field
        py::array_t<double> scalar_field(num_ver);
        auto scalar_ptr = scalar_field.mutable_data();

        for (int i = 0; i < num_ver && i < data[col_index].size(); i++) {
            scalar_ptr[i] = data[col_index][i];
        }

        sol[field_name.c_str()] = scalar_field;
        col_index++;
    }

    return sol;
}

bool write_solution(const std::string& solpath, py::dict sol, int num_ver, int dim) {
    if (!pymeshb::createDirectory(solpath)) {
        return false;
    }

    std::ofstream sol_file(solpath);
    if (!sol_file.is_open()) {
        return false;
    }

    // Prepare data vectors and column headers
    std::vector<std::string> headers;
    std::vector<std::vector<double>> data_columns;

    // Process each field
    for (auto item : sol) {
        std::string key = item.first.cast<std::string>();
        py::array_t<double> field = item.second.cast<py::array_t<double>>();

        if (field.ndim() == 1) {
            // Scalar field
            headers.push_back("\"" + key + "\"");
            std::vector<double> column(num_ver);
            auto field_ptr = field.data();
            for (int i = 0; i < num_ver; i++) {
                column[i] = field_ptr[i];
            }
            data_columns.push_back(column);
        }
        else if (field.ndim() == 2 && field.shape(1) == dim) {
            // Vector field
            auto field_ptr = field.data();
            for (int j = 0; j < dim; j++) {
                char component = 'X' + j;
                headers.push_back("\"" + key + "_" + component + "\"");
                std::vector<double> column(num_ver);
                for (int i = 0; i < num_ver; i++) {
                    column[i] = field_ptr[i * dim + j];
                }
                data_columns.push_back(column);
            }
        }
        else if (field.ndim() == 2 && field.shape(1) == (dim * (dim + 1)) / 2) {
            // Symmetric tensor field
            auto field_ptr = field.data();
            int sym_size = (dim * (dim + 1)) / 2;
            std::vector<std::string> components;

            if (dim == 2) {
                components = {"XX", "XY", "YY"};
            } else {
                components = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
            }

            for (int j = 0; j < sym_size; j++) {
                headers.push_back("\"" + key + "_" + components[j] + "\"");
                std::vector<double> column(num_ver);
                for (int i = 0; i < num_ver; i++) {
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
        for (int i = 0; i < num_ver; i++) {
            sol_file << data_columns[0][i];
            for (size_t j = 1; j < data_columns.size(); j++) {
                sol_file << "," << data_columns[j][i];
            }
            sol_file << std::endl;
        }
    }

    sol_file.close();
    return true;
}

}  // namespace su2
}  // namespace pymeshb