#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "ref_map.hpp"

namespace pymeshb {
namespace su2 {

std::map<int, std::string> RefMap::loadRefMap(const std::string& filename) {
    std::map<int, std::string> ref_map;

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
                        ref_map[marker_id] = marker_name;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // If loading fails, return an empty map
        ref_map.clear();
    }

    return ref_map;
}

void RefMap::saveRefMap(const std::map<int, std::string>& ref_map, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# SU2 Marker Map\n";
        file << "# Format: marker_id:marker_name\n";

        for (const auto& [key, value] : ref_map) {
            file << key << ":" << value << std::endl;
        }
    }
}

std::string RefMap::getMarkerName(const std::map<int, std::string>& ref_map, int marker_id) {
    auto it = ref_map.find(marker_id);
    if (it != ref_map.end()) {
        return it->second;
    }
    return "MARKER_" + std::to_string(marker_id);
}

} // namespace su2
} // namespace pymeshb