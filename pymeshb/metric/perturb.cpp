/** @file perturb.cpp
 *  @brief Function implementations for metric tensor perturbations.
 *
 *  Implementation for tensor perturbation operations on individual tensors and fields.
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

// Forward declarations
py::tuple decompose(py::array_t<double> lower_tri);
py::array_t<double> recompose(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors);

/**
 * Apply perturbation to eigenvalues and eigenvectors and compute perturbed tensor
 *
 * @param eigenvalues Original eigenvalues
 * @param eigenvectors Original eigenvectors
 * @param delta_eigenvals Perturbation values for eigenvalues
 * @param rotation_angles Rotational perturbation values for eigenvectors
 * @return Tuple containing perturbed eigenvalues and eigenvectors
 */
py::tuple perturb(py::array_t<double> eigenvalues,
                  py::array_t<double> eigenvectors,
                  py::array_t<double> delta_eigenvals,
                  py::array_t<double> rotation_angles) {
    auto eig_vals = eigenvalues.unchecked<1>();
    auto eig_vecs = eigenvectors.unchecked<2>();
    auto delta_vals = delta_eigenvals.unchecked<1>();
    auto rot_angles = rotation_angles.unchecked<1>();

    int dim = eig_vals.shape(0);
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Dimension must be 1, 2, or 3");
    }

    // Validate input dimensions
    if (eig_vecs.shape(0) != dim || eig_vecs.shape(1) != dim) {
        throw std::runtime_error("Eigenvector dimensions must match eigenvalue count");
    }
    if (delta_vals.shape(0) != dim) {
        throw std::runtime_error("Eigenvalue perturbations must match eigenvalue count");
    }

    int num_angle = (dim == 1) ? 0 : ((dim == 2) ? 1 : 3);
    if (rot_angles.shape(0) != num_angle) {
        throw std::runtime_error("Invalid rotation angles array size: "
            "should be 0 for 1D, 1 for 2D, or 3 for 3D");
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
        p_vals(i) = std::exp(std::log(eigen_val) + delta_vals(i));
    }

    // Perturb eigenvectors
    // First, copy the original eigenvectors as the starting point
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            p_vecs(i, j) = eig_vecs(i, j);
        }
    }

    // Apply rotational perturbations based on dimension
    if (dim == 2) {
        // In 2D, we only need one rotation angle
        double theta = rot_angles(0);

        // Create eigenvector matrix
        Eigen::Matrix2d V;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                V(i, j) = eig_vecs(i, j);
            }
        }

        // Create rotation matrix
        Eigen::Matrix2d rot;
        rot << std::cos(theta), -std::sin(theta),
               std::sin(theta), std::cos(theta);

        // Apply rotation
        Eigen::Matrix2d V_rotated = V * rot;

        // Copy back to output
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                p_vecs(i, j) = V_rotated(i, j);
            }
        }
    }
    else if (dim == 3) {
        // In 3D, we need three rotation angles for xy, yz, xz planes
        double theta_xy = rot_angles(0);  // rotation in xy plane
        double theta_yz = rot_angles(1);  // rotation in yz plane
        double theta_xz = rot_angles(2);  // rotation in xz plane

        // Create eigenvector matrix
        Eigen::Matrix3d V;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                V(i, j) = eig_vecs(i, j);
            }
        }

        // Create rotation matrices for each plane
        Eigen::Matrix3d R_xy = Eigen::Matrix3d::Identity();
        R_xy(0, 0) = std::cos(theta_xy);
        R_xy(0, 1) = -std::sin(theta_xy);
        R_xy(1, 0) = std::sin(theta_xy);
        R_xy(1, 1) = std::cos(theta_xy);

        Eigen::Matrix3d R_yz = Eigen::Matrix3d::Identity();
        R_yz(1, 1) = std::cos(theta_yz);
        R_yz(1, 2) = -std::sin(theta_yz);
        R_yz(2, 1) = std::sin(theta_yz);
        R_yz(2, 2) = std::cos(theta_yz);

        Eigen::Matrix3d R_xz = Eigen::Matrix3d::Identity();
        R_xz(0, 0) = std::cos(theta_xz);
        R_xz(0, 2) = std::sin(theta_xz);
        R_xz(2, 0) = -std::sin(theta_xz);
        R_xz(2, 2) = std::cos(theta_xz);

        // Apply all rotations (order matters)
        Eigen::Matrix3d R_combined = R_xy * R_yz * R_xz;
        Eigen::Matrix3d V_rotated = V * R_combined;

        // Copy back to output
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                p_vecs(i, j) = V_rotated(i, j);
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
 * @param delta_eigenvals Array of shape (num_point, dim) containing
 *                        eigenvalue perturbations for each point
 * @param rotation_angles Array of shape (num_point, num_angle) containing
 *                        eigenvector rotational perturbations for each point
 * @return Array of perturbed metric tensors in lower triangular form
 */
py::array_t<double> perturb_metric_field(
    py::array_t<double> metrics,
    py::array_t<double> delta_eigenvals,
    py::array_t<double> rotation_angles) {

    auto metrics_info = metrics.request();
    auto delta_vals_info = delta_eigenvals.request();
    auto rot_angles_info = rotation_angles.request();

    unsigned int num_point = metrics_info.shape[0];
    unsigned int num_met = metrics_info.shape[1];
    int dim = 0;
    if (num_met == 1) dim = 1;
    else if (num_met == 3) dim = 2;
    else if (num_met == 6) dim = 3;
    else throw std::runtime_error("Invalid tensor size: must be 1, 3, or 6 elements per tensor");

    // Validate input shapes
    if (metrics_info.ndim != 2) {
        throw std::runtime_error("Tensors array must be 2-dimensional");
    }
    if (delta_vals_info.ndim != 2 || delta_vals_info.shape[0] != num_point ||
        delta_vals_info.shape[1] != dim) {
        throw std::runtime_error("Eigenvalue perturbations shape must be (num_point, dim)");
    }

    int num_angle = (dim == 1) ? 0 : ((dim == 2) ? 1 : 3);
    if (rot_angles_info.ndim != 2 || rot_angles_info.shape[0] != num_point ||
        rot_angles_info.shape[1] != num_angle) {
        throw std::runtime_error("Rotation angles shape must be (num_point, num_angle) "
                                 "where num_angle is 0 for 1D, 1 for 2D, or 3 for 3D");
    }

    // Create output array
    py::array_t<double> perturbed_metrics({num_point, num_met});
    auto result_info = perturbed_metrics.request();

    // Direct pointers to data
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);
    double* delta_vals_ptr = static_cast<double*>(delta_vals_info.ptr);
    double* rot_angles_ptr = static_cast<double*>(rot_angles_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each tensor
    for (auto i = 0; i < num_point; i++) {
        // Create view for current tensor
        py::array_t<double> metric({num_met}, {sizeof(double)},
                                   metrics_ptr + i * num_met);

        // Diagonalize
        py::tuple diag_result = decompose(metric);
        py::array_t<double> eigenvalues = diag_result[0].cast<py::array_t<double>>();
        py::array_t<double> eigenvectors = diag_result[1].cast<py::array_t<double>>();

        // Create views for current perturbations
        py::array_t<double> delta_vals({dim}, {sizeof(double)},
                                       delta_vals_ptr + i * dim);
        py::array_t<double> rot_angles({num_angle}, {sizeof(double)},
                                       rot_angles_ptr + i * num_angle);

        // Apply perturbations
        py::tuple pert_result = perturb(eigenvalues, eigenvectors, delta_vals, rot_angles);
        py::array_t<double> perturbed_vals = pert_result[0].cast<py::array_t<double>>();
        py::array_t<double> perturbed_vecs = pert_result[1].cast<py::array_t<double>>();

        // Recombine
        py::array_t<double> perturbed_tensor = recompose(perturbed_vals, perturbed_vecs);

        // Copy to output array
        auto perturbed_info = perturbed_tensor.request();
        double* perturbed_ptr = static_cast<double*>(perturbed_info.ptr);
        std::memcpy(result_ptr + i * num_met, perturbed_ptr, num_met * sizeof(double));
    }

    return perturbed_metrics;
}
