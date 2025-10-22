#ifndef LIMON_MESH_REFINE_3D_HPP
#define LIMON_MESH_REFINE_3D_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace limon {
namespace mesh {

py::dict refine_3d(const py::dict& mesh_data);

} // namespace mesh
} // namespace limon

#endif // LIMON_MESH_REFINE_3D_HPP
