/** @file metric.cpp
 *  @brief Function implementations for metric tensor operations.
 *
 *  Implementations for tensor operations including diagonalization,
 *  recombination, and perturbation of symmetric tensors.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

/**
 * Diagonalize a symmetric tensor represented by its lower triangular elements
 *
 * @param lower_tri Array containing the lower triangular elements of tensor
 * @return Tuple containing eigenvalues and eigenvectors
 */
py::tuple diagonalize(py::array_t<double> lower_tri) {
    auto buf = lower_tri.request();
    double *ptr = static_cast<double *>(buf.ptr);

    // Determine dimension from lower triangle size
    int dim = 0;
    if (buf.size == 1) dim = 1;
    else if (buf.size == 3) dim = 2;
    else if (buf.size == 6) dim = 3;
    else throw std::runtime_error("Input size must be 1, 3, or 6 for 1D, 2D, or 3D tensor");

    // Allocate return arrays
    py::array_t<double> eigenvalues(dim);
    auto eig_vals = eigenvalues.mutable_unchecked<1>();

    py::array_t<double> eigenvectors({dim, dim});
    auto eig_vecs = eigenvectors.mutable_unchecked<2>();

    // 1D case is trivial
    if (dim == 1) {
        eig_vals(0) = ptr[0];
        eig_vecs(0, 0) = 1.0;
        return py::make_tuple(eigenvalues, eigenvectors);
    }

    // 2D case
    else if (dim == 2) {
        double a = ptr[0]; // a[0][0]
        double b = ptr[1]; // a[1][0]
        double c = ptr[2]; // a[1][1]

        // Calculate eigenvalues using quadratic formula
        double trace = a + c;
        double det = a * c - b * b;
        double discriminant = trace * trace - 4 * det;

        // Check for numerical issues
        if (discriminant < 0 && discriminant > -1e-10) {
            discriminant = 0;
        } else if (discriminant < 0) {
            throw std::runtime_error("Negative discriminant in 2D eigenvalue calculation");
        }

        eig_vals(0) = (trace + std::sqrt(discriminant)) / 2.0;
        eig_vals(1) = (trace - std::sqrt(discriminant)) / 2.0;

        // Calculate eigenvectors
        for (auto i = 0; i < 2; i++) {
            double lambda = eig_vals(i);

            if (std::abs(b) > 1e-10) {
                double v1 = lambda - c;
                double v2 = b;
                double norm = std::sqrt(v1*v1 + v2*v2);

                eig_vecs(0, i) = v1 / norm;
                eig_vecs(1, i) = v2 / norm;
            } else {
                // b is effectively zero, matrix is already diagonal
                if (std::abs(lambda - a) < 1e-10) {
                    // First eigenvector
                    eig_vecs(0, i) = 1.0;
                    eig_vecs(1, i) = 0.0;
                } else {
                    // Second eigenvector
                    eig_vecs(0, i) = 0.0;
                    eig_vecs(1, i) = 1.0;
                }
            }
        }

        return py::make_tuple(eigenvalues, eigenvectors);
    }

    // 3D case
    else {
        // Construct the full symmetric matrix
        double matrix[3][3] = {
            {ptr[0], ptr[1], ptr[3]},
            {ptr[1], ptr[2], ptr[4]},
            {ptr[3], ptr[4], ptr[5]}
        };

        // Initialize eigenvectors as identity matrix
        double evecs[3][3] = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        // Make working copy of matrix
        double a[3][3];
        for (auto i = 0; i < 3; i++) {
            for (auto j = 0; j < 3; j++) {
                a[i][j] = matrix[i][j];
            }
        }

        // Jacobi iteration to find eigenvalues and eigenvectors
        const int MAX_ITER = 50;
        const double EPSILON = 1e-10;

        for (auto iter = 0; iter < MAX_ITER; iter++) {
            // Find largest off-diagonal element
            int p = 0, q = 1;
            double max_val = std::abs(a[p][q]);

            for (auto i = 0; i < 3; i++) {
                for (auto j = i+1; j < 3; j++) {
                    if (std::abs(a[i][j]) > max_val) {
                        max_val = std::abs(a[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }

            // Check for convergence
            if (max_val < EPSILON) break;

            // Calculate rotation angles
            double theta = 0.5 * atan2(2 * a[p][q], a[p][p] - a[q][q]);
            double c = cos(theta);
            double s = sin(theta);

            // Apply rotation
            double app = a[p][p];
            double aqq = a[q][q];
            double apq = a[p][q];

            a[p][p] = app * c * c + aqq * s * s + 2 * apq * c * s;
            a[q][q] = app * s * s + aqq * c * c - 2 * apq * c * s;
            a[p][q] = a[q][p] = 0.0;  // Should be zero after rotation

            for (auto i = 0; i < 3; i++) {
                if (i != p && i != q) {
                    double api = a[p][i];
                    double aqi = a[q][i];
                    a[p][i] = a[i][p] = api * c + aqi * s;
                    a[q][i] = a[i][q] = -api * s + aqi * c;
                }
            }

            // Update eigenvectors
            for (auto i = 0; i < 3; i++) {
                double vip = evecs[i][p];
                double viq = evecs[i][q];
                evecs[i][p] = vip * c + viq * s;
                evecs[i][q] = -vip * s + viq * c;
            }
        }

        // Copy results into return arrays
        for (auto i = 0; i < 3; i++) {
            eig_vals(i) = a[i][i];
        }

        // Sort eigenvalues and eigenvectors (descending order)
        for (auto i = 0; i < 3; i++) {
            for (auto j = i + 1; j < 3; j++) {
                if (eig_vals(i) < eig_vals(j)) {
                    // Swap eigenvalues
                    double temp = eig_vals(i);
                    eig_vals(i) = eig_vals(j);
                    eig_vals(j) = temp;

                    // Swap eigenvectors
                    for (auto k = 0; k < 3; k++) {
                        temp = evecs[k][i];
                        evecs[k][i] = evecs[k][j];
                        evecs[k][j] = temp;
                    }
                }
            }
        }

        // Copy eigenvectors to output array
        for (auto i = 0; i < 3; i++) {
            for (auto j = 0; j < 3; j++) {
                eig_vecs(i, j) = evecs[i][j];
            }
        }

        return py::make_tuple(eigenvalues, eigenvectors);
    }
}

/**
 * Recombine eigenvalues and eigenvectors to reconstruct the tensor
 *
 * @param eigenvalues Array of eigenvalues
 * @param eigenvectors Array of eigenvectors (column-wise)
 * @return Lower triangular elements of reconstructed tensor
 */
py::array_t<double> recombine(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors) {
    auto eig_vals = eigenvalues.unchecked<1>();
    auto eig_vecs = eigenvectors.unchecked<2>();

    int dim = eig_vals.shape(0);
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Dimension must be 1, 2, or 3");
    }

    if (eig_vecs.shape(0) != dim || eig_vecs.shape(1) != dim) {
        throw std::runtime_error("Eigenvector dimensions must match eigenvalue count");
    }

    // Output will be the lower triangular elements
    int num_elements = dim * (dim + 1) / 2;
    py::array_t<double> lower_tri(num_elements);
    auto result = lower_tri.mutable_unchecked<1>();

    // Full matrix reconstruction: M = V * D * V^T
    double matrix[3][3] = {{0}};

    for (auto i = 0; i < dim; i++) {
        for (auto j = 0; j < dim; j++) {
            double sum = 0.0;
            for (auto k = 0; k < dim; k++) {
                sum += eig_vecs(i, k) * eig_vals(k) * eig_vecs(j, k);
            }
            matrix[i][j] = sum;
        }
    }

    // Copy to result (lower triangular elements only)
    if (dim == 1) {
        result(0) = matrix[0][0];
    } else if (dim == 2) {
        result(0) = matrix[0][0]; // a[0][0]
        result(1) = matrix[1][0]; // a[1][0]
        result(2) = matrix[1][1]; // a[1][1]
    } else { // dim == 3
        result(0) = matrix[0][0]; // a[0][0]
        result(1) = matrix[1][0]; // a[1][0]
        result(2) = matrix[1][1]; // a[1][1]
        result(3) = matrix[2][0]; // a[2][0]
        result(4) = matrix[2][1]; // a[2][1]
        result(5) = matrix[2][2]; // a[2][2]
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