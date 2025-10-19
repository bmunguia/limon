/** @file integrate.cpp
 *  @brief Function implementation for metric field integration.
 *
 *  Implementation for integration operations on metric tensor fields.
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
 * Integrate the determinant of metric tensors over volumes
 *
 * For each tensor in the input array, this function:
 * 1. Reconstructs the full tensor from lower triangular elements
 * 2. Computes the determinant of the tensor
 * 3. Multiplies by the corresponding volume and adds to the integral
 *
 * @param metrics Array of shape (num_point, num_met) where each row
 *                contains the lower triangular elements of a tensor
 * @param volumes Array of shape (num_point,) containing volumes for each point
 * @param norm Choice of norm for Lp-norm normalization
 * @return Scalar value representing the integrated determinant
 */
double integrate_metric_field(
    py::array_t<double> metrics,
    py::array_t<double> volumes,
    int norm) {

    auto metrics_info = metrics.request();
    auto volumes_info = volumes.request();

    unsigned int num_point = metrics_info.shape[0];
    unsigned int num_met = metrics_info.shape[1];
    int dim = 0;
    if (num_met == 1) dim = 1;
    else if (num_met == 3) dim = 2;
    else if (num_met == 6) dim = 3;
    else throw std::runtime_error("Invalid tensor size: must be 1, 3, or 6 elements per tensor");

    // Validate input shapes
    if (metrics_info.ndim != 2) {
        throw std::runtime_error("Metrics array must be 2D");
    }
    if (volumes_info.ndim != 1 || volumes_info.shape[0] != num_point) {
        throw std::runtime_error("Volumes shape must be (num_point,)");
    }

    // Direct pointers to data
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);
    double* volumes_ptr = static_cast<double*>(volumes_info.ptr);

    double integral = 0.0;

    // Exponent for global normalization term
    double norm_exp = norm / (2.0 * norm + dim);

    // Pre-allocate matrix
    Eigen::MatrixXd A(dim, dim);

    // Process each tensor
    for (unsigned int i = 0; i < num_point; i++) {
        // Reconstruct full tensor matrix from lower triangular elements
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

        // Get volume for this point
        double volume = volumes_ptr[i];

        // Add to integral
        integral += std::pow(det, norm_exp) * volume;
    }

    return integral;
}
