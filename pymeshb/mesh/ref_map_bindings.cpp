#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ref_map.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ref_map, m) {
    m.doc() = "Reference map utilities for mesh markers and solution labels";

    py::enum_<pymeshb::RefMapKind>(m, "RefMapKind")
        .value("Marker", pymeshb::RefMapKind::Marker)
        .value("Solution", pymeshb::RefMapKind::Solution);

    m.def("load_ref_map", &pymeshb::RefMap::loadRefMap,
          py::arg("filename"),
          "Load reference map from file");
    m.def("write_ref_map", &pymeshb::RefMap::writeRefMap,
          py::arg("ref_map"),
          py::arg("filename"),
          py::arg("kind"),
          "Write reference map to file");
    m.def("get_ref_name", &pymeshb::RefMap::getRefName,
          py::arg("ref_map"),
          py::arg("ref_id"),
          "Get reference name from map");
}
