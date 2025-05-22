#include <filesystem>
#include <fstream>

#include "ref_map.hpp"

namespace pymeshb {

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

                    // Parse line as "ref_id:ref_name"
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        int ref_id = std::stoi(line.substr(0, pos));
                        std::string ref_name = line.substr(pos + 1);
                        ref_map[ref_id] = ref_name;
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
        file << "# SU2 Reference ID Map\n";
        file << "# Format: ref_id:ref_name\n";

        for (const auto& [key, value] : ref_map) {
            file << key << ":" << value << std::endl;
        }
    }
}

std::string RefMap::getRefName(const std::map<int, std::string>& ref_map, int ref_id) {
    auto it = ref_map.find(ref_id);
    if (it != ref_map.end()) {
        return it->second;
    }
    return "REF_" + std::to_string(ref_id);
}

} // namespace pymeshb