/** @file euler_angle.cpp
 *  @brief Function implementations for rotation angle operations.
 *
 *  Implementation for converting eigenvectors to rotation angles.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Convert the eingenvectors to rotation angles
 *
 * @param eigenvectors Eigenvectors
 * @return Array of shape (1,) for 2D and (3,) for 3D containing rotation angles
 */
py::array_t<double> rotation_angles(py::array_t<double> eigenvectors) {
    auto eigenvecs_info = eigenvectors.request();

    // Validate input shapes
    if (eigenvecs_info.ndim != 2) {
        throw std::invalid_argument("Eigenvectors must be a 2D array.");
    }
    size_t dim = eigenvecs_info.shape[0];
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument("Only 2D and 3D eigenvectors are supported.");
    }

    Eigen::Map<Eigen::MatrixXd> eigen_mat(static_cast<double *>(eigenvecs_info.ptr), eigenvecs_info.shape[0], eigenvecs_info.shape[1]);

    if (dim == 2) {
        // 2D case: Compute angle using Eigen's rotation matrix
        Eigen::Rotation2Dd rotation(eigen_mat.col(0));
        double angle = rotation.angle();
        return py::array_t<double>({1}, {angle});
    } else {
        // 3D case: Compute Euler angles (ZYX order)
        Eigen::Matrix3d rotation_matrix = eigen_mat;
        Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order
        return py::array_t<double>({3}, {euler_angles[0], euler_angles[1], euler_angles[2]});
    }
}

/**
 * Convert the field of eigenvectors to rotation angles
 *
 * @param eigenvectors Eigenvectors
 * @return Array of shape (num_point, 1) for 2D and (num_point, 3) for 3D containing rotation angles
 */
py::array_t<double> rotation_angles_field(py::array_t<double> eigenvectors) {
    auto eigenvecs_info = eigenvectors.request();

    // Validate input shapes
    if (eigenvecs_info.ndim != 3) {
        throw std::invalid_argument("Eigenvectors array must be 3D.");
    }
    size_t num_point = eigenvecs_info.shape[0];
    size_t dim = eigenvecs_info.shape[1];
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument("Only 2D and 3D eigenvectors are supported.");
    }

    // Create output array
    size_t output_dim = (dim == 2) ? 1 : 3;
    py::array_t<double> angles(std::vector<py::ssize_t>{num_point, output_dim});
    auto result_info = angles.request();

    // Direct pointers to data
    double* eigenvecs_ptr = static_cast<double*>(eigenvecs_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each eigenvector array
    for (size_t i = 0; i < num_point; ++i) {
        // Create views for current eigenvectors
        py::array_t<double> current_eigenvector(std::vector<py::ssize_t>{dim, dim},
                                               std::vector<py::ssize_t>{dim * sizeof(double), sizeof(double)},
                                               eigenvecs_ptr + i * dim * dim);

        // Rotation angles for the current eigenvector
        py::array_t<double> vector_angle = rotation_angles(current_eigenvector);

        // Copy to output array
        auto vec_angle_info = vector_angle.request();
        double* vec_angle_ptr = static_cast<double*>(vec_angle_info.ptr);
        std::memcpy(result_ptr + i * output_dim, vec_angle_ptr, output_dim * sizeof(double));
    }

    return result;
}