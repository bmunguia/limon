#ifndef LIMON_GEOMETRY_SURFACE_METRIC_H
#define LIMON_GEOMETRY_SURFACE_METRIC_H

#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace limon {
namespace geometry {

/**
 * Surface-geometry metric of a mesh: boundary nodes get a tangent size set by the local
 * curvature, every other node the isotropic size hmax.
 *
 * 2-D port of SU2's geometricSurfaceMetrics (tag phd-greenlight,
 * SU2_CFD/include/metrics/computeMetrics.hpp).
 *
 * @param mesh_data Mesh dictionary with coords, boundaries['Edges'] (last column: marker id),
 *                  optional boundaries['Corners'], markers ({id: name}) and dim
 * @param geodev Ordered (marker name, deviation) pairs; a node on several markers takes the first
 * @param mode "angle" (deviation in degrees) or "hausdorff" (sagitta distance)
 * @param hmin Smallest edge length
 * @param hmax Largest edge length
 * @return (num_point, 3) C-contiguous array of [xx, xy, yy]
 */
py::array_t<double> surface_metric(const py::dict& mesh_data,
                                   const std::vector<std::pair<std::string, double>>& geodev,
                                   const std::string& mode, double hmin, double hmax);

} // namespace geometry
} // namespace limon

#endif // LIMON_GEOMETRY_SURFACE_METRIC_H
