#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ref_map.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ref_map, m) {
    m.doc() = "Reference map utilities for mesh markers and solution labels";

    py::enum_<limon::RefMapKind>(m, "RefMapKind")
        .value("Marker", limon::RefMapKind::Marker)
        .value("Solution", limon::RefMapKind::Solution);

    m.def("load_ref_map", &limon::RefMap::loadRefMap,
          py::arg("filename"),
          "Load reference map from file");
    m.def("write_ref_map", &limon::RefMap::writeRefMap,
          py::arg("ref_map"),
          py::arg("filename"),
          py::arg("kind"),
          "Write reference map to file");
    m.def("get_ref_name", &limon::RefMap::getRefName,
          py::arg("ref_map"),
          py::arg("ref_id"),
          "Get reference name from map");
}
