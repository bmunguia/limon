#ifndef LIMON_MESHUTIL_H
#define LIMON_MESHUTIL_H

#include <string>

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
}

#endif // LIMON_MESHUTIL_H