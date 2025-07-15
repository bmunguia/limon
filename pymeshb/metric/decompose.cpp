/** @file decompose.cpp
 *  @brief Function implementations for metric tensor decomposition.
 *
 *  Implementation for tensor diagonalization and recombination operations.
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
 * Diagonalize a symmetric tensor represented by its lower triangular elements using Eigen
 *
 * @param lower_tri Array containing the lower triangular elements of tensor
 * @return Tuple containing eigenvalues and eigenvectors
 */
py::tuple decompose(py::array_t<double> lower_tri) {
    auto buf = lower_tri.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input lower_tri must be a 1D array.");
    }
    double *ptr = static_cast<double *>(buf.ptr);

    int dim = 0;
    if (buf.size == 1) dim = 1;
    else if (buf.size == 3) dim = 2;
    else if (buf.size == 6) dim = 3;
    else throw std::runtime_error("Input size must be 1, 3, or 6 for 1D, 2D, or 3D tensor");

    Eigen::MatrixXd A(dim, dim);

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

    // Use SelfAdjointEigenSolver since metric is symmetric positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed.");
    }

    Eigen::VectorXd eigen_values_eigen = es.eigenvalues();
    Eigen::MatrixXd eigen_vectors_eigen = es.eigenvectors();

    // Prepare pybind11 output arrays
    py::array_t<double> eigenvalues(dim);
    auto eig_vals_ptr = eigenvalues.mutable_unchecked<1>();

    py::array_t<double> eigenvectors(std::vector<py::ssize_t>{dim, dim});
    auto eig_vecs_ptr = eigenvectors.mutable_unchecked<2>();

    // Eigen returns eigenvalues in ascending order. We need descending.
    // Create pairs of (eigenvalue, eigenvector_column) to sort
    std::vector<std::pair<double, Eigen::VectorXd>> eigen_pairs(dim);
    for (int i = 0; i < dim; ++i) {
        eigen_pairs[i] = {eigen_values_eigen(i), eigen_vectors_eigen.col(i)};
    }

    // Sort in descending order of eigenvalues
    std::sort(eigen_pairs.begin(), eigen_pairs.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });

    // Populate the pybind arrays
    for (int i = 0; i < dim; ++i) {
        eig_vals_ptr(i) = eigen_pairs[i].first;
        for (int j = 0; j < dim; ++j) {
            // Store j-th component of i-th (sorted) eigenvector
            // Eigenvectors are columns in eigenvectors
            eig_vecs_ptr(j, i) = eigen_pairs[i].second(j);
        }
    }

    return py::make_tuple(eigenvalues, eigenvectors);
}

/**
 * Decompose a field of metric tensors
 *
 * For each tensor in the input array, this function:
 * 1. Diagonalizes the tensor into eigenvalues and eigenvectors
 * 2. Returns the eigenvalues and eigenvectors for all points
 *
 * @param metrics Array of shape (num_point, num_met) where each row
 *                contains the lower triangular elements of a tensor
 * @return Tuple containing eigenvalues array of shape (num_point, dim) and
 *         eigenvectors array of shape (num_point, dim, dim)
 */
py::tuple decompose_metric_field(py::array_t<double> metrics) {
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

    // Create output arrays
    py::array_t<double> all_eigenvalues(std::vector<py::ssize_t>{num_point, dim});
    py::array_t<double> all_eigenvectors(std::vector<py::ssize_t>{num_point, dim, dim});
    auto eigenvals_info = all_eigenvalues.request();
    auto eigenvecs_info = all_eigenvectors.request();

    // Direct pointers to data
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);
    double* eigenvals_ptr = static_cast<double*>(eigenvals_info.ptr);
    double* eigenvecs_ptr = static_cast<double*>(eigenvecs_info.ptr);

    // Process each tensor
    for (unsigned int i = 0; i < num_point; i++) {
        // Create view for current tensor
        py::array_t<double> metric(std::vector<py::ssize_t>{num_met}, std::vector<py::ssize_t>{sizeof(double)},
                                   metrics_ptr + i * num_met);

        // Decompose current tensor
        py::tuple result = decompose(metric);
        py::array_t<double> eigenvalues = result[0].cast<py::array_t<double>>();
        py::array_t<double> eigenvectors = result[1].cast<py::array_t<double>>();

        // Copy eigenvalues to output array
        auto eig_vals_info = eigenvalues.request();
        double* eig_vals_ptr = static_cast<double*>(eig_vals_info.ptr);
        std::memcpy(eigenvals_ptr + i * dim, eig_vals_ptr, dim * sizeof(double));

        // Copy eigenvectors to output array
        auto eig_vecs_info = eigenvectors.request();
        double* eig_vecs_ptr = static_cast<double*>(eig_vecs_info.ptr);
        std::memcpy(eigenvecs_ptr + i * dim * dim, eig_vecs_ptr, dim * dim * sizeof(double));
    }

    return py::make_tuple(all_eigenvalues, all_eigenvectors);
}
