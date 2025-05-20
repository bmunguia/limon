#pragma once

#include <string>
#include <map>
#include <fstream>
#include <stdexcept>

namespace pymeshb {
namespace su2 {

class MarkerMap {
public:
    static std::map<int, std::string> loadMarkerMap(const std::string& filename);

    static void saveMarkerMap(const std::map<int, std::string>& marker_map, const std::string& filename);

    static std::string getMarkerName(const std::map<int, std::string>& marker_map, int marker_id);
};

} // namespace su2
} // namespace pymeshb