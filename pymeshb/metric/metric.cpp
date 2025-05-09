/** @file metric.cpp
 *  @brief Function implementations for metric tensor operations.
 *
 *  Implementations for tensor operations including diagonalization,
 *  recombination, and perturbation of symmetric tensors.
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
py::tuple diagonalize(py::array_t<double> lower_tri) {
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

    py::array_t<double> eigenvectors({(ptrdiff_t)dim, (ptrdiff_t)dim});
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
 * Recombine eigenvalues and eigenvectors to reconstruct the tensor using Eigen
 *
 * @param eigenvalues Array of eigenvalues
 * @param eigenvectors Array of eigenvectors (column-wise)
 * @return Lower triangular elements of reconstructed tensor
 */
py::array_t<double> recombine(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors) {
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
 * Apply perturbation to eigenvalues and eigenvectors and compute perturbed tensor
 *
 * @param eigenvalues Original eigenvalues
 * @param eigenvectors Original eigenvectors
 * @param val_perturbations Perturbation values for eigenvalues
 * @param vec_perturbations Perturbation values for eigenvectors
 * @return Tuple containing perturbed eigenvalues and eigenvectors
 */
py::tuple perturb(py::array_t<double> eigenvalues,
                  py::array_t<double> eigenvectors,
                  py::array_t<double> val_perturbations,
                  py::array_t<double> vec_perturbations) {
    auto eig_vals = eigenvalues.unchecked<1>();
    auto eig_vecs = eigenvectors.unchecked<2>();
    auto val_pert = val_perturbations.unchecked<1>();
    auto vec_pert = vec_perturbations.unchecked<2>();

    int dim = eig_vals.shape(0);
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Dimension must be 1, 2, or 3");
    }

    // Validate input dimensions
    if (eig_vecs.shape(0) != dim || eig_vecs.shape(1) != dim) {
        throw std::runtime_error("Eigenvector dimensions must match eigenvalue count");
    }
    if (val_pert.shape(0) != dim) {
        throw std::runtime_error("Eigenvalue perturbations must match eigenvalue count");
    }
    if (vec_pert.shape(0) != dim || vec_pert.shape(1) != dim) {
        throw std::runtime_error("Eigenvector perturbations dimensions must match eigenvectors");
    }

    // Apply perturbations
    py::array_t<double> perturbed_vals(dim);
    py::array_t<double> perturbed_vecs({dim, dim});
    auto p_vals = perturbed_vals.mutable_unchecked<1>();
    auto p_vecs = perturbed_vecs.mutable_unchecked<2>();

    // Perturb eigenvalues
    for (auto i = 0; i < dim; i++) {
        // Ensure eigenvalues are positive before taking log
        double eigen_val = std::max(eig_vals(i), 1e-10);
        // Take log, perturb, then exponentiate
        p_vals(i) = std::exp(std::log(eigen_val) + val_pert(i));
    }

    // Perturb eigenvectors
    for (auto i = 0; i < dim; i++) {
        for (auto j = 0; j < dim; j++) {
            p_vecs(i, j) = eig_vecs(i, j) + vec_pert(i, j);
        }
    }

    // Orthonormalize eigenvectors using Gram-Schmidt
    for (auto j = 0; j < dim; j++) {
        // Normalize the j-th vector
        double norm = 0.0;
        for (auto i = 0; i < dim; i++) {
            norm += p_vecs(i, j) * p_vecs(i, j);
        }
        norm = std::sqrt(norm);

        for (auto i = 0; i < dim; i++) {
            p_vecs(i, j) /= norm;
        }

        // Orthogonalize against previous vectors
        for (auto k = j + 1; k < dim; k++) {
            double dot = 0.0;
            for (auto i = 0; i < dim; i++) {
                dot += p_vecs(i, j) * p_vecs(i, k);
            }

            for (auto i = 0; i < dim; i++) {
                p_vecs(i, k) -= dot * p_vecs(i, j);
            }
        }
    }

    return py::make_tuple(perturbed_vals, perturbed_vecs);
}

/**
 * Perturb a field of metric tensors
 *
 * For each tensor in the input array, this function:
 * 1. Diagonalizes the tensor into eigenvalues and eigenvectors
 * 2. Applies specified perturbations to the eigenvalues and eigenvectors
 * 3. Recombines into a perturbed tensor
 *
 * @param metrics Array of shape (num_point, num_met) where each row
 *                contains the lower triangular elements of a tensor
 * @param val_perturbations Array of shape (num_point, dim) containing
 *                        eigenvalue perturbations for each point
 * @param vec_perturbations Array of shape (num_point, dim, dim) containing
 *                        eigenvector perturbations for each point
 * @return Array of perturbed metric tensors in lower triangular form
 */
py::array_t<double> perturb_metric_field(
    py::array_t<double> metrics,
    py::array_t<double> val_perturbations,
    py::array_t<double> vec_perturbations) {

    // Get input array info
    auto metrics_info = metrics.request();
    auto val_pert_info = val_perturbations.request();
    auto vec_pert_info = vec_perturbations.request();

    // Check dimensions
    if (metrics_info.ndim != 2) {
        throw std::runtime_error("Tensors array must be 2-dimensional");
    }

    int64_t num_point = metrics_info.shape[0];
    int64_t num_met = metrics_info.shape[1];

    // Determine dimension from lower triangle size
    int dim = 0;
    if (num_met == 1) dim = 1;
    else if (num_met == 3) dim = 2;
    else if (num_met == 6) dim = 3;
    else throw std::runtime_error("Invalid tensor size: must be 1, 3, or 6 elements per tensor");

    // Validate input shapes
    if (val_pert_info.ndim != 2 || val_pert_info.shape[0] != num_point ||
        val_pert_info.shape[1] != dim) {
        throw std::runtime_error("Eigenvalue perturbations shape must be (num_point, dim)");
    }

    if (vec_pert_info.ndim != 3 || vec_pert_info.shape[0] != num_point ||
        vec_pert_info.shape[1] != dim || vec_pert_info.shape[2] != dim) {
        throw std::runtime_error("Eigenvector perturbations shape must be (num_point, dim, dim)");
    }

    // Create output array
    py::array_t<double> perturbed_metrics({num_point, num_met});
    auto result_info = perturbed_metrics.request();

    // Direct pointers to data
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);
    double* val_pert_ptr = static_cast<double*>(val_pert_info.ptr);
    double* vec_pert_ptr = static_cast<double*>(vec_pert_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each tensor
    for (auto i = 0; i < num_point; i++) {
        // Create view for current tensor
        py::array_t<double> metric({num_met}, {sizeof(double)},
                                   metrics_ptr + i * num_met);

        // Diagonalize
        py::tuple diag_result = diagonalize(metric);
        py::array_t<double> eigenvalues = diag_result[0].cast<py::array_t<double>>();
        py::array_t<double> eigenvectors = diag_result[1].cast<py::array_t<double>>();

        // Create views for current perturbations
        py::array_t<double> val_pert({dim}, {sizeof(double)},
                                    val_pert_ptr + i * dim);
        py::array_t<double> vec_pert({dim, dim}, {dim * sizeof(double), sizeof(double)},
                                    vec_pert_ptr + i * dim * dim);

        // Apply perturbations
        py::tuple pert_result = perturb(eigenvalues, eigenvectors, val_pert, vec_pert);
        py::array_t<double> perturbed_vals = pert_result[0].cast<py::array_t<double>>();
        py::array_t<double> perturbed_vecs = pert_result[1].cast<py::array_t<double>>();

        // Recombine
        py::array_t<double> perturbed_tensor = recombine(perturbed_vals, perturbed_vecs);

        // Copy to output array
        auto perturbed_info = perturbed_tensor.request();
        double* perturbed_ptr = static_cast<double*>(perturbed_info.ptr);
        std::memcpy(result_ptr + i * num_met, perturbed_ptr, num_met * sizeof(double));
    }

    return perturbed_metrics;
}

/**
 * Python module definition for metric tensor operations.
 */
PYBIND11_MODULE(_metric, m) {
    m.doc() = "Metric tensor operations";

    m.def("diagonalize", &diagonalize, "Diagonalize a symmetric tensor",
          py::arg("lower_tri"));

    m.def("recombine", &recombine, "Recombine eigenvalues and eigenvectors to reconstruct the tensor",
          py::arg("eigenvalues"), py::arg("eigenvectors"));

    m.def("perturb", &perturb, "Apply perturbation to eigenvalues and eigenvectors",
          py::arg("eigenvalues"), py::arg("eigenvectors"),
          py::arg("val_perturbations"), py::arg("vec_perturbations"));

    m.def("perturb_metric_field", &perturb_metric_field,
          "Perturb a field of metric tensors",
          py::arg("metrics"), py::arg("val_perturbations"), py::arg("vec_perturbations"));
}