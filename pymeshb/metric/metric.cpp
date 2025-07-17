/** @file metric.cpp
 *  @brief Python module definition for metric tensor operations.
 *
 *  This file contains the pybind11 module definition that exposes
 *  all metric tensor operations to Python.
 *
 *  @author Brian Munguía
 *  @bug No known bugs.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of functions defined in other files
py::tuple decompose(py::array_t<double> lower_tri);
py::array_t<double> recompose(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors);
py::tuple decompose_metric_field(py::array_t<double> metrics);
py::array_t<double> recompose_metric_field(py::array_t<double> eigenvalues, py::array_t<double> eigenvectors);
py::tuple perturb(py::array_t<double> eigenvalues,
                  py::array_t<double> eigenvectors,
                  py::array_t<double> delta_eigenvals,
                  py::array_t<double> rotation_angles);
py::array_t<double> perturb_metric_field(
    py::array_t<double> metrics,
    py::array_t<double> delta_eigenvals,
    py::array_t<double> rotation_angles);
double integrate_metric_field(
    py::array_t<double> metrics,
    py::array_t<double> volumes,
    int norm);
py::array_t<double> normalize_metric_field(
    py::array_t<double> metrics,
    double metric_integral,
    int complexity,
    int norm);
py::array_t<double> metric_edge_length_at_endpoints(
    py::array_t<int> edges,
    py::array_t<double> coords,
    py::array_t<double> metrics);
py::array_t<double> metric_edge_length(
    py::array_t<int> edges,
    py::array_t<double> coords,
    py::array_t<double> metrics,
    double eps = 1e-12);
py::array_t<double> rotation_angles(py::array_t<double> eigenvectors);
py::array_t<double> rotation_angles_field(py::array_t<double> eigenvectors);

/**
 * Python module definition for metric tensor operations.
 */
PYBIND11_MODULE(_metric, m) {
    m.doc() = "Metric tensor operations";

    m.def("decompose", &decompose,
        "Diagonalize a symmetric tensor",
        py::arg("lower_tri"));

    m.def("recompose", &recompose,
        "Recombine eigenvalues and eigenvectors to reconstruct the tensor",
        py::arg("eigenvalues"), py::arg("eigenvectors"));

    m.def("decompose_metric_field", &decompose_metric_field,
        "Decompose a field of metric tensors",
        py::arg("metrics"));

    m.def("recompose_metric_field", &recompose_metric_field,
        "Recompose a field of metric tensors from eigenvalues and eigenvectors",
        py::arg("eigenvalues"), py::arg("eigenvectors"));

    m.def("perturb", &perturb,
        "Apply perturbation to eigenvalues and eigenvectors",
        py::arg("eigenvalues"), py::arg("eigenvectors"),
        py::arg("delta_eigenvals"), py::arg("rotation_angles"));

    m.def("perturb_metric_field", &perturb_metric_field,
          "Perturb a field of metric tensors",
          py::arg("metrics"), py::arg("delta_eigenvals"), py::arg("rotation_angles"));

    m.def("integrate_metric_field", &integrate_metric_field,
          "Integrate determinant of metric tensors over volumes",
          py::arg("metrics"), py::arg("volumes"), py::arg("norm"));

    m.def("normalize_metric_field", &normalize_metric_field,
          "Normalize a field of metric tensors",
          py::arg("metrics"), py::arg("metric_integral"),
          py::arg("complexity"), py::arg("norm"));

    m.def("metric_edge_length_at_endpoints", &metric_edge_length_at_endpoints,
          "Compute the metric edge length at the endpoints of the edges",
          py::arg("edges"), py::arg("coords"), py::arg("metrics"));

    m.def("metric_edge_length", &metric_edge_length,
          "Compute the metric edge length for the given edges",
          py::arg("edges"), py::arg("coords"), py::arg("metrics"),
          py::arg("eps") = 1e-12);

    m.def("rotation_angles", &rotation_angles,
          "Compute rotation angles from eigenvectors",
          py::arg("eigenvectors"));

    m.def("rotation_angles_field", &rotation_angles_field,
          "Compute rotation angles for a field of eigenvectors",
          py::arg("eigenvectors"));
}