#ifndef PYMESHB_MESHUTIL_H
#define PYMESHB_MESHUTIL_H

#include <string>

namespace pymeshb {
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

#endif // PYMESHB_MESHUTIL_H