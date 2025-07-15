/** @file edge_length.cpp
 *  @brief Function implementations for edge length calculation in metric space.
 *
 *  Implementation for Riemannian metric edge length calculations based on
 *  the geometric integration approach by Alauzet, Loseille, et al. at Inria.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Helper function to compute vector norm in Riemannian metric space
 * Calculates sqrt(v^T * M * v) where M is reconstructed from lower triangular elements
 *
 * @param edge_vector Edge vector (coordinates difference)
 * @param metric_tril Lower triangular metric elements
 * @param dim Dimension (1, 2, or 3)
 * @return Metric norm of the vector
 */
double metric_vector_norm(const Eigen::VectorXd& edge_vector,
                         const double* metric_tril,
                         int dim) {
    // Reconstruct the full metric tensor from lower triangular elements
    Eigen::MatrixXd M(dim, dim);

    if (dim == 1) {
        M(0,0) = metric_tril[0];
    } else if (dim == 2) {
        // lower_tri: [M00, M10, M11]
        M(0,0) = metric_tril[0];
        M(1,0) = metric_tril[1]; M(0,1) = metric_tril[1];
        M(1,1) = metric_tril[2];
    } else { // dim == 3
        // lower_tri: [M00, M10, M11, M20, M21, M22]
        M(0,0) = metric_tril[0];
        M(1,0) = metric_tril[1]; M(0,1) = metric_tril[1];
        M(1,1) = metric_tril[2];
        M(2,0) = metric_tril[3]; M(0,2) = metric_tril[3];
        M(2,1) = metric_tril[4]; M(1,2) = metric_tril[4];
        M(2,2) = metric_tril[5];
    }

    // Compute v^T * M * v
    Eigen::VectorXd Mv = M * edge_vector;
    double norm_squared = edge_vector.dot(Mv);

    return std::sqrt(norm_squared);
}

/**
 * Calculate edge length in Riemannian metric space evaluated at endpoints
 *
 * For each edge, computes the metric length ||e||_M at both endpoints,
 * where e is the edge vector and M is the metric tensor.
 *
 * @param edges Array of shape (num_edge, 2) containing node indices for each edge
 * @param coords Array of shape (num_node, dim) containing node coordinates
 * @param metrics Array of shape (num_node, num_met) containing metric tensors
 *                at each node in lower triangular form
 * @return Array of shape (num_edge, 2) containing metric lengths at both endpoints
 */
py::array_t<double> metric_edge_length_at_endpoints(
    py::array_t<int> edges,
    py::array_t<double> coords,
    py::array_t<double> metrics) {

    auto edges_info = edges.request();
    auto coords_info = coords.request();
    auto metrics_info = metrics.request();

    // Validate input shapes
    if (edges_info.ndim != 2 || edges_info.shape[1] != 2) {
        throw std::runtime_error("Edges array must be 2-dimensional with shape (num_edge, 2)");
    }
    if (coords_info.ndim != 2) {
        throw std::runtime_error("Coordinates array must be 2-dimensional");
    }
    if (metrics_info.ndim != 2) {
        throw std::runtime_error("Metrics array must be 2-dimensional");
    }

    unsigned int num_edge = edges_info.shape[0];
    unsigned int num_node = coords_info.shape[0];
    unsigned int dim = coords_info.shape[1];
    unsigned int num_met = metrics_info.shape[1];

    // Validate dimension consistency
    if (metrics_info.shape[0] != num_node) {
        throw std::runtime_error("Metrics array must have same number of rows as coordinates");
    }

    // Determine tensor dimension from number of metric elements
    int expected_num_met = 0;
    if (dim == 1) expected_num_met = 1;
    else if (dim == 2) expected_num_met = 3;
    else if (dim == 3) expected_num_met = 6;
    else throw std::runtime_error("Dimension must be 1, 2, or 3");

    if (num_met != expected_num_met) {
        throw std::runtime_error("Number of metric elements must match dimension: "
                                 "1 for 1D, 3 for 2D, 6 for 3D");
    }

    // Create output array
    py::array_t<double> edge_lengths(std::vector<py::ssize_t>{num_edge, 2});
    auto result_info = edge_lengths.request();

    // Direct pointers to data
    int* edges_ptr = static_cast<int*>(edges_info.ptr);
    double* coords_ptr = static_cast<double*>(coords_info.ptr);
    double* metrics_ptr = static_cast<double*>(metrics_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each edge
    for (unsigned int e = 0; e < num_edge; e++) {
        int node_a = edges_ptr[e * 2];
        int node_b = edges_ptr[e * 2 + 1];

        // Validate node indices
        if (node_a < 0 || node_a >= static_cast<int>(num_node) ||
            node_b < 0 || node_b >= static_cast<int>(num_node)) {
            throw std::runtime_error("Edge contains invalid node index");
        }

        // Compute edge vector
        Eigen::VectorXd edge_vector(dim);
        for (unsigned int d = 0; d < dim; d++) {
            edge_vector(d) = coords_ptr[node_b * dim + d] - coords_ptr[node_a * dim + d];
        }

        // Compute metric length at both endpoints
        double* metric_a = metrics_ptr + node_a * num_met;
        double* metric_b = metrics_ptr + node_b * num_met;

        result_ptr[e * 2] = metric_vector_norm(edge_vector, metric_a, dim);
        result_ptr[e * 2 + 1] = metric_vector_norm(edge_vector, metric_b, dim);
    }

    return edge_lengths;
}

/**
 * Calculate integrated edge length in Riemannian metric space
 *
 * Uses geometric integration approach based on Alauzet, Loseille, et al.
 * For each edge, computes the integrated metric length using the formula:
 * - If ratio r ≈ 1: l_M = l_a
 * - Otherwise: l_M = l_a * (r - 1) / ln(r)
 * where r = l_b / l_a and l_a, l_b are metric lengths at endpoints.
 *
 * @param edges Array of shape (num_edge, 2) containing node indices for each edge
 * @param coords Array of shape (num_node, dim) containing node coordinates
 * @param metrics Array of shape (num_node, num_met) containing metric tensors
 *                at each node in lower triangular form
 * @param eps Tolerance for ratio comparison (default: 1e-12)
 * @return Array of shape (num_edge,) containing integrated metric lengths
 */
py::array_t<double> metric_edge_length(
    py::array_t<int> edges,
    py::array_t<double> coords,
    py::array_t<double> metrics,
    double eps = 1e-12) {

    // First get edge lengths at endpoints
    py::array_t<double> l_ab = metric_edge_length_at_endpoints(edges, coords, metrics);
    auto l_ab_info = l_ab.request();

    unsigned int num_edge = l_ab_info.shape[0];

    // Create output array
    py::array_t<double> integrated_lengths(std::vector<py::ssize_t>{num_edge});
    auto result_info = integrated_lengths.request();

    // Direct pointers to data
    double* l_ab_ptr = static_cast<double*>(l_ab_info.ptr);
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Process each edge
    for (unsigned int e = 0; e < num_edge; e++) {
        double l_a = l_ab_ptr[e * 2];
        double l_b = l_ab_ptr[e * 2 + 1];

        // Handle degenerate cases
        if (l_a <= 0.0) {
            result_ptr[e] = l_b;
            continue;
        }
        if (l_b <= 0.0) {
            result_ptr[e] = l_a;
            continue;
        }

        // Compute ratio
        double r = l_b / l_a;

        // Approximate edge length in metric space
        if (std::abs(r - 1.0) <= eps) {
            // Ratio is close to 1, use l_a
            result_ptr[e] = l_a;
        } else {
            // Integrated length along edge: l_a * (r - 1) / ln(r)
            result_ptr[e] = l_a * (r - 1.0) / std::log(r);
        }
    }

    return integrated_lengths;
}
