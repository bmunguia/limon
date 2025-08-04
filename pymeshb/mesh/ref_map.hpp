#pragma once

#include <string>
#include <map>
#include <fstream>
#include <stdexcept>

namespace pymeshb {

enum class RefMapKind {
    Marker = 0,
    Solution = 1
};

class RefMap {
public:
    static std::map<int, std::string> loadRefMap(const std::string& filename);

    static void writeRefMap(const std::map<int, std::string>& ref_map, const std::string& filename, RefMapKind kind);

    static std::string getRefName(const std::map<int, std::string>& ref_map, int marker_id);
};

} // namespace pymeshb