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
#include <cstring>
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

    // Use row-major storage to match numpy's default layout
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_mat(
        static_cast<double *>(eigenvecs_info.ptr), eigenvecs_info.shape[0], eigenvecs_info.shape[1]);

    if (dim == 2) {
        // 2D case: Extract angle from first eigenvector (first column)
        // The angle is atan2(y, x) where (x, y) is the first eigenvector
        Eigen::Vector2d first_eigenvec = eigen_mat.col(0);
        double angle = std::atan2(first_eigenvec(1), first_eigenvec(0));

        // Create result array with proper initialization
        auto result = py::array_t<double>(1);
        auto result_info = result.request();
        static_cast<double*>(result_info.ptr)[0] = angle;

        return result;
    } else {
        // 3D case: Compute Euler angles (ZYX order)
        Eigen::Matrix3d rotation_matrix = eigen_mat;
        Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order

        // Create result array with proper initialization
        auto result = py::array_t<double>(3);
        py::buffer_info buf_info = result.request();
        double* ptr = static_cast<double*>(buf_info.ptr);
        ptr[0] = euler_angles[0];
        ptr[1] = euler_angles[1];
        ptr[2] = euler_angles[2];

        return result;
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
    auto angles = py::array_t<double>({static_cast<py::ssize_t>(num_point), static_cast<py::ssize_t>(output_dim)});
    auto angles_info = angles.request();

    // Direct pointers to data
    double* eigenvecs_ptr = static_cast<double*>(eigenvecs_info.ptr);
    double* angles_ptr = static_cast<double*>(angles_info.ptr);

    // Process each eigenvector array
    for (size_t i = 0; i < num_point; ++i) {
        // Create a view for the current eigenvector matrix
        py::array_t<double> current_eigenvecs = py::array_t<double>(
            {static_cast<py::ssize_t>(dim), static_cast<py::ssize_t>(dim)},
            {static_cast<py::ssize_t>(dim * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
            eigenvecs_ptr + i * dim * dim,
            py::handle(eigenvectors)
        );

        // Call rotation_angles for this point
        py::array_t<double> point_angles = rotation_angles(current_eigenvecs);

        // Copy results to output array
        auto point_angles_info = point_angles.request();
        double* point_angles_ptr = static_cast<double*>(point_angles_info.ptr);
        std::memcpy(angles_ptr + i * output_dim, point_angles_ptr, output_dim * sizeof(double));
    }

    return angles;
}