#ifndef LIMON_MESHUTIL_H
#define LIMON_MESHUTIL_H

#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace limon {
    /**
     * Remove leading and trailing whitespace from a string.
     *
     * @param str The input string to trim
     * @return The trimmed string
     */
    std::string trim(const std::string& str);

    /**
     * Create directory structure for a file path if it doesn't exist.
     *
     * @param path Path to create directories for
     * @return true if successful, false otherwise
     */
    bool createDirectory(const std::string& path);

    /**
     * Cast a Python object to a C-contiguous array of T.
     *
     * Writers index the buffer as row-major, so a transposed or sliced view must
     * be copied; arrays that are already C-contiguous are borrowed without a copy.
     *
     * @param h Python object convertible to an array
     * @return C-contiguous array of T
     */
    template <typename T>
    py::array_t<T> contiguous(const py::handle& h) {
        auto arr = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
        if (!arr) {
            throw std::runtime_error("expected a numeric array");
        }
        return py::reinterpret_borrow<py::array_t<T>>(arr);
    }
}

#endif // LIMON_MESHUTIL_H