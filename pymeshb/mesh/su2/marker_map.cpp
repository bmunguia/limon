#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "marker_map.hpp"

namespace pymeshb {
namespace su2 {

std::map<int, std::string> MarkerMap::loadMarkerMap(const std::string& filename) {
    std::map<int, std::string> marker_map;

    try {
        if (std::filesystem::exists(filename)) {
            std::ifstream file(filename);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    // Skip empty lines and comments
                    if (line.empty() || line[0] == '#' || line[0] == '/') {
                        continue;
                    }

                    // Parse line as "marker_id:marker_name"
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        int marker_id = std::stoi(line.substr(0, pos));
                        std::string marker_name = line.substr(pos + 1);
                        marker_map[marker_id] = marker_name;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // If loading fails, return an empty map
        marker_map.clear();
    }

    return marker_map;
}

void MarkerMap::saveMarkerMap(const std::map<int, std::string>& marker_map, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# SU2 Marker Map\n";
        file << "# Format: marker_id:marker_name\n";

        for (const auto& [key, value] : marker_map) {
            file << key << ":" << value << std::endl;
        }
    }
}

std::string MarkerMap::getMarkerName(const std::map<int, std::string>& marker_map, int marker_id) {
    auto it = marker_map.find(marker_id);
    if (it != marker_map.end()) {
        return it->second;
    }
    return "MARKER_" + std::to_string(marker_id);
}

} // namespace su2
} // namespace pymeshb