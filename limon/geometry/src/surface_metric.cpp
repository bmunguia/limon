#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "../../mesh/common/include/util.hpp"
#include "../include/surface_metric.hpp"

namespace limon {
namespace geometry {

namespace {

constexpr double kCurvatureFloor = 1e-24;
constexpr double kPi = 3.14159265358979323846;

/** Signed curvature of the circle through three points (Menger curvature). */
double menger_curvature(const double* p0, const double* p1, const double* p2) {
    const double ax = p1[0] - p0[0], ay = p1[1] - p0[1];
    const double bx = p2[0] - p0[0], by = p2[1] - p0[1];
    const double cross = ax * by - ay * bx;
    const double l10 = std::hypot(ax, ay);
    const double l21 = std::hypot(p2[0] - p1[0], p2[1] - p1[1]);
    const double l20 = std::hypot(bx, by);
    return 2.0 * cross / (l10 * l21 * l20 + kCurvatureFloor);
}

/** Target edge length for a curvature, before clipping to [hmin, hmax]. */
double geo_size(double deviation, double abs_curvature, bool angle_mode, double hmax) {
    if (deviation <= 0.0) {
        return hmax;
    }
    const double k = std::max(abs_curvature, kCurvatureFloor);
    if (angle_mode) {
        return deviation * kPi / 180.0 / k;
    }
    return std::sqrt(8.0 * deviation / k);
}

} // namespace

py::array_t<double> surface_metric(const py::dict& mesh_data,
                                   const std::vector<std::pair<std::string, double>>& geodev,
                                   const std::string& mode, double hmin, double hmax) {
    if (mode != "angle" && mode != "hausdorff") {
        throw std::invalid_argument("mode must be 'angle' or 'hausdorff', got '" + mode + "'");
    }
    if (!(hmin > 0.0) || !(hmax > hmin)) {
        throw std::invalid_argument("sizes must satisfy 0 < hmin < hmax");
    }
    const int dim = mesh_data["dim"].cast<int>();
    if (dim != 2) {
        throw std::runtime_error(
            "surface_metric is implemented for 2-D meshes only (3-D: see SU2 tag phd-greenlight, "
            "SU2_CFD/include/metrics/computeMetrics.hpp)");
    }
    const bool angle_mode = (mode == "angle");

    py::array_t<double> coords = limon::contiguous<double>(mesh_data["coords"]);
    const double* xy = coords.data();
    const py::ssize_t num_point = coords.shape(0);

    py::dict boundaries = mesh_data["boundaries"].cast<py::dict>();
    if (!boundaries.contains("Edges")) {
        throw std::invalid_argument("mesh_data['boundaries'] has no 'Edges'");
    }
    py::array_t<unsigned int> edges = limon::contiguous<unsigned int>(boundaries["Edges"]);
    const unsigned int* edge_ptr = edges.data();
    const py::ssize_t num_edge = edges.shape(0);
    const py::ssize_t edge_width = edges.shape(1);

    std::unordered_set<unsigned int> corners;
    if (boundaries.contains("Corners")) {
        py::array_t<unsigned int> corner_array = limon::contiguous<unsigned int>(boundaries["Corners"]);
        for (py::ssize_t i = 0; i < corner_array.size(); i++) {
            corners.insert(corner_array.data()[i]);
        }
    }

    std::unordered_map<std::string, int> marker_id;
    if (mesh_data.contains("markers")) {
        for (auto item : mesh_data["markers"].cast<py::dict>()) {
            marker_id[item.second.cast<std::string>()] = item.first.cast<int>();
        }
    }

    // Every node starts isotropic at the largest size
    const double eigmin = 1.0 / (hmax * hmax);
    py::array_t<double> metric(std::vector<py::ssize_t>{num_point, 3});
    double* m = metric.mutable_data();
    for (py::ssize_t i = 0; i < num_point; i++) {
        m[3 * i + 0] = eigmin;
        m[3 * i + 1] = 0.0;
        m[3 * i + 2] = eigmin;
    }

    std::vector<char> processed(num_point, 0);

    for (const auto& [name, deviation] : geodev) {
        auto found = marker_id.find(name);
        if (found == marker_id.end()) {
            throw std::invalid_argument("geodev marker '" + name + "' is not in mesh_data['markers']");
        }

        // Neighbors along this marker's boundary edges, nodes in order of first appearance
        std::vector<unsigned int> nodes;
        std::unordered_map<unsigned int, std::vector<unsigned int>> neighbors;
        for (py::ssize_t e = 0; e < num_edge; e++) {
            const unsigned int* row = edge_ptr + e * edge_width;
            if (static_cast<int>(row[edge_width - 1]) != found->second) {
                continue;
            }
            for (int end = 0; end < 2; end++) {
                const unsigned int a = row[end], b = row[1 - end];
                if (!neighbors.count(a)) {
                    nodes.push_back(a);
                }
                neighbors[a].push_back(b);
            }
        }

        for (const unsigned int node : nodes) {
            if (corners.count(node) || processed[node]) {
                continue;
            }
            processed[node] = 1;

            // Curvature is defined where the boundary passes through the node once
            const auto& nb = neighbors[node];
            if (nb.size() != 2) {
                continue;
            }
            const double* prev = xy + 2 * nb[0];
            const double* curr = xy + 2 * node;
            const double* next = xy + 2 * nb[1];

            const double kappa = menger_curvature(prev, curr, next);
            const double h = std::clamp(geo_size(deviation, std::fabs(kappa), angle_mode, hmax), hmin, hmax);

            // Tangent follows the chord between the neighbors (the boundary normal's perpendicular)
            double tx = next[0] - prev[0], ty = next[1] - prev[1];
            const double length = std::hypot(tx, ty);
            if (length == 0.0) {
                continue;
            }
            tx /= length;
            ty /= length;

            const double extra = 1.0 / (h * h) - eigmin;
            m[3 * node + 0] = eigmin + extra * tx * tx;
            m[3 * node + 1] = extra * tx * ty;
            m[3 * node + 2] = eigmin + extra * ty * ty;
        }
    }

    return metric;
}

} // namespace geometry
} // namespace limon
