/** @file normalize.cpp
 *  @brief Function implementation for metric field normalization.
 *
 *  Implementation for normalization operations on metric tensor fields.
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
 * Normalize a field of metric tensors
 *
 * For each tensor in the input array, this function:
 * 1. Reconstructs the full tensor from lower triangular elements
 * 2. Computes the determinant of the tensor
 * 3. Scales the tensor by the normalization factor
 *
 * @param metrics Array of shape (num_point, num_met) where each row
 *                contains the lower triangular elements of a tensor
 * @param metric_integral Integral of metric determinants over the domain
 * @param complexity Target complexity for normalization
 * @param norm Choice of norm for Lp-norm normalization
 * @return Array of normalized metric tensors in lower triangular form
 */
py::array_t<double> normalize_metric_field(
    py::array_t<double> metrics,
    double metric_integral,
    int complexity,
    int norm) {

    auto metrics_info = metrics.request();

    unsigned int num_point = metrics_info.shape[0];
    unsigned int num_met = metrics_info.shape[1];
    int dim = 0;
    if (num_met == 1) dim = 1;
    else if (num_met == 3) dim = 2;
    else if (num_met == 6) dim = 3;
    else throw std::runtime_error("Invalid tensor size: must be 1, 3, or 6 elements per tensor");

    // Validate input shapes
    if (metrics_info.ndim != 2) {
        throw std::runtime_error("Metrics array must be 2-dimensional");
    }

    // Direct pointers to data
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);

    // Create output array
    py::array_t<double> normalized_metrics(std::vector<py::ssize_t>{num_point, num_met});
    auto result_info = normalized_metrics.request();
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Global scaling factor
    double global_scale = std::pow(complexity / metric_integral, 2.0 / dim);

    // Process each tensor
    for (unsigned int i = 0; i < num_point; i++) {
        // Reconstruct full tensor matrix from lower triangular elements
        Eigen::MatrixXd A(dim, dim);
        double* ptr = metrics_ptr + i * num_met;

        if (dim == 1) {
            A(0,0) = ptr[0];
        } else if (dim == 2) {
            // lower_tri: [A00, A10, A11]
            A(0,0) = ptr[0];
            A(1,0) = ptr[1]; A(0,1) = ptr[1];
            A(1,1) = ptr[2];
        } else { // dim == 3
            // lower_tri: [A00, A10, A11, A20, A21, A22]
            A(0,0) = ptr[0];
            A(1,0) = ptr[1]; A(0,1) = ptr[1];
            A(1,1) = ptr[2];
            A(2,0) = ptr[3]; A(0,2) = ptr[3];
            A(2,1) = ptr[4]; A(1,2) = ptr[4];
            A(2,2) = ptr[5];
        }

        // Compute determinant
        double det = A.determinant();

        // Local scaling factor
        double local_scale = std::pow(det, -1.0 / (2.0 * norm + dim));

        // Total scaling factor
        double total_scale = global_scale * local_scale;

        // Scale the tensor
        Eigen::MatrixXd A_scaled = total_scale * A;

        // Extract lower triangular elements and store in output
        double* result_tensor_ptr = result_ptr + i * num_met;
        if (dim == 1) {
            result_tensor_ptr[0] = A_scaled(0,0);
        } else if (dim == 2) {
            result_tensor_ptr[0] = A_scaled(0,0);
            result_tensor_ptr[1] = A_scaled(1,0);
            result_tensor_ptr[2] = A_scaled(1,1);
        } else { // dim == 3
            result_tensor_ptr[0] = A_scaled(0,0);
            result_tensor_ptr[1] = A_scaled(1,0);
            result_tensor_ptr[2] = A_scaled(1,1);
            result_tensor_ptr[3] = A_scaled(2,0);
            result_tensor_ptr[4] = A_scaled(2,1);
            result_tensor_ptr[5] = A_scaled(2,2);
        }
    }

    return normalized_metrics;
}
