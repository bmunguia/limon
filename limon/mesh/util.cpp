#include "util.hpp"
#include <cstdlib>

namespace limon {
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, (last - first + 1));
    }

    bool createDirectory(const std::string& path) {
        size_t lastSlash = path.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string dir = path.substr(0, lastSlash);
            std::string cmd = "mkdir -p " + dir;
            int ret = system(cmd.c_str());
            return ret == 0;
        }
        return true;
    }
}