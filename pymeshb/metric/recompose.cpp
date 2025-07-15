/** @file recompose.cpp
 *  @brief Function implementation for metric tensor recomposition.
 *
 *  Implementation for reconstructing tensors from eigenvalues and eigenvectors.
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
 * Recombine eigenvalues and eigenvectors to reconstruct the tensor using Eigen
 *
 * @param eigenvalues Array of eigenvalues
 * @param eigenvectors Array of eigenvectors (column-wise)
 * @return Lower triangular elements of reconstructed tensor
 */
py::array_t<double> recompose(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors) {
    auto eig_vals_info = eigenvalues.request();
    auto eig_vecs_info = eigenvectors.request();

    if (eig_vals_info.ndim != 1) {
        throw std::runtime_error("Eigenvalues must be a 1D array.");
    }
    if (eig_vecs_info.ndim != 2) {
        throw std::runtime_error("Eigenvectors must be a 2D array.");
    }

    int dim = eig_vals_info.shape[0];
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Dimension must be 1, 2, or 3");
    }

    if (eig_vecs_info.shape[0] != dim || eig_vecs_info.shape[1] != dim) {
        throw std::runtime_error("Eigenvector dimensions must match eigenvalue count and be square.");
    }

    // Convert py::array_t to Eigen types
    Eigen::VectorXd D_vec(dim);
    auto eig_vals_unchecked = eigenvalues.unchecked<1>();
    for (int i = 0; i < dim; ++i) {
        D_vec(i) = eig_vals_unchecked(i);
    }

    Eigen::MatrixXd V(dim, dim);
    auto eig_vecs_unchecked = eigenvectors.unchecked<2>();
    for (int i = 0; i < dim; ++i) { // rows
        for (int j = 0; j < dim; ++j) { // cols
            // Assuming eigenvectors are columns in the input py::array_t
            // V(row_idx, col_idx_of_eigenvector)
            V(i, j) = eig_vecs_unchecked(i, j);
        }
    }

    // Reconstruct: M = V * D_diag * V^T
    Eigen::MatrixXd D_diag = D_vec.asDiagonal();
    Eigen::MatrixXd M = V * D_diag * V.transpose();

    // Output will be the lower triangular elements
    int num_elements = dim * (dim + 1) / 2;
    py::array_t<double> lower_tri(num_elements);
    auto result_ptr = lower_tri.mutable_unchecked<1>();

    if (dim == 1) {
        result_ptr(0) = M(0,0);
    } else if (dim == 2) {
        result_ptr(0) = M(0,0); // M(0,0)
        result_ptr(1) = M(1,0); // M(1,0)
        result_ptr(2) = M(1,1); // M(1,1)
    } else { // dim == 3
        result_ptr(0) = M(0,0); // M(0,0)
        result_ptr(1) = M(1,0); // M(1,0)
        result_ptr(2) = M(1,1); // M(1,1)
        result_ptr(3) = M(2,0); // M(2,0)
        result_ptr(4) = M(2,1); // M(2,1)
        result_ptr(5) = M(2,2); // M(2,2)
    }

    return lower_tri;
}

/**
 * Recompose a field of metric tensors from eigenvalues and eigenvectors
 *
 * For each set of eigenvalues and eigenvectors in the input arrays, this function:
 * 1. Reconstructs the tensor from eigenvalues and eigenvectors
 * 2. Returns the lower triangular elements for all points
 *
 * @param eigenvalues Array of shape (num_point, dim) containing eigenvalues for each point
 * @param eigenvectors Array of shape (num_point, dim, dim) containing eigenvectors for each point
 * @return Array of shape (num_point, num_met) containing recomposed metric tensors
 */
py::array_t<double> recompose_metric_field(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors) {
    auto eigenvals_info = eigenvalues.request();
    auto eigenvecs_info = eigenvectors.request();

    // Validate input shapes
    if (eigenvals_info.ndim != 2) {
        throw std::runtime_error("Eigenvalues array must be 2-dimensional");
    }
    if (eigenvecs_info.ndim != 3) {
        throw std::runtime_error("Eigenvectors array must be 3-dimensional");
    }

    unsigned int num_point = eigenvals_info.shape[0];
    unsigned int dim = eigenvals_info.shape[1];

    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Dimension must be 1, 2, or 3");
    }

    if (eigenvecs_info.shape[0] != num_point ||
        eigenvecs_info.shape[1] != dim ||
        eigenvecs_info.shape[2] != dim) {
        throw std::runtime_error("Eigenvectors shape must be (num_point, dim, dim)");
    }

    // Calculate output tensor size
    unsigned int num_met = dim * (dim + 1) / 2;

    // Create output array
    py::array_t<double> recomposed_metrics(std::vector<py::ssize_t>{num_point, num_met});
    auto result_info = recomposed_metrics.request();

    // Direct pointers to data
    double* eigenvals_ptr = static_cast<double*>(eigenvals_info.ptr);
    double* eigenvecs_ptr = static_cast<double*>(eigenvecs_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each tensor
    for (unsigned int i = 0; i < num_point; i++) {
        // Create views for current eigenvalues and eigenvectors
        py::array_t<double> current_eigenvals(std::vector<py::ssize_t>{dim}, std::vector<py::ssize_t>{sizeof(double)},
                                              eigenvals_ptr + i * dim);
        py::array_t<double> current_eigenvecs(std::vector<py::ssize_t>{dim, dim}, std::vector<py::ssize_t>{dim * sizeof(double), sizeof(double)},
                                              eigenvecs_ptr + i * dim * dim);

        // Recompose current tensor
        py::array_t<double> recomposed_tensor = recompose(current_eigenvals, current_eigenvecs);

        // Copy to output array
        auto tensor_info = recomposed_tensor.request();
        double* tensor_ptr = static_cast<double*>(tensor_info.ptr);
        std::memcpy(result_ptr + i * num_met, tensor_ptr, num_met * sizeof(double));
    }

    return recomposed_metrics;
}
