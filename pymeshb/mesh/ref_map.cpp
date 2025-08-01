#include <filesystem>
#include <fstream>

#include "ref_map.hpp"

namespace pymeshb {

std::map<int, std::string> RefMap::loadRefMap(const std::string& filename) {
    std::map<int, std::string> ref_map;
    try {
        if (!std::filesystem::exists(filename)) {
            return ref_map;
        }
        std::ifstream file(filename);
        if (!file.is_open()) {
            return ref_map;
        }

        std::string ext;
        size_t dot = filename.find_last_of('.');
        if (dot != std::string::npos) {
            ext = filename.substr(dot);
        }

        if (ext == ".csv") {
            std::string line;
            // Skip header
            std::getline(file, line);
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                size_t comma = line.find(',');
                if (comma != std::string::npos) {
                    int ref_id = std::stoi(line.substr(0, comma));
                    std::string ref_name = line.substr(comma + 1);
                    // Remove leading/trailing spaces and quotes
                    ref_name.erase(0, ref_name.find_first_not_of(" \"'"));
                    ref_name.erase(ref_name.find_last_not_of(" \"'") + 1);
                    ref_map[ref_id] = ref_name;
                }
            }
        } else if (ext == ".dat") {
            std::string line;
            // Skip Tecplot headers (TITLE, VARIABLES, ZONE)
            for (int i = 0; i < 3; ++i) std::getline(file, line);
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                size_t comma = line.find(',');
                if (comma != std::string::npos) {
                    int ref_id = std::stoi(line.substr(0, comma));
                    std::string ref_name = line.substr(comma + 1);
                    ref_name.erase(0, ref_name.find_first_not_of(" \"'"));
                    ref_name.erase(ref_name.find_last_not_of(" \"'") + 1);
                    ref_map[ref_id] = ref_name;
                }
            }
        } else if (ext == ".json") {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            // Remove braces
            size_t start = content.find('{');
            size_t end = content.find('}');
            if (start == std::string::npos || end == std::string::npos) return ref_map;
            std::string dict = content.substr(start + 1, end - start - 1);
            size_t pos = 0;
            while (pos < dict.size()) {
                // Find key
                size_t key_start = dict.find('"', pos);
                if (key_start == std::string::npos) key_start = dict.find('\'', pos);
                if (key_start == std::string::npos) break;
                size_t key_end = dict.find_first_of("'\"", key_start + 1);
                if (key_end == std::string::npos) break;
                std::string key = dict.substr(key_start + 1, key_end - key_start - 1);
                // Find colon
                size_t colon = dict.find(':', key_end);
                if (colon == std::string::npos) break;
                // Find value
                size_t value_start = dict.find_first_not_of(" ", colon + 1);
                if (value_start == std::string::npos) break;
                size_t value_end = dict.find_first_of(",}", value_start);
                std::string value = dict.substr(value_start, value_end - value_start);
                int ref_id = std::stoi(value);
                ref_map[ref_id] = key;
                pos = value_end;
            }
        } else {
            throw std::runtime_error("Unsupported file extension for ref map: " + ext);
        }
    } catch (const std::exception& e) {
        // If loading fails, return an empty map
        ref_map.clear();
    }

    return ref_map;
}

void RefMap::saveRefMap(const std::map<int, std::string>& ref_map, const std::string& filename, RefMapKind kind) {
    std::string ext;
    size_t dot = filename.find_last_of('.');
    if (dot != std::string::npos) {
        ext = filename.substr(dot);
    }

    std::ofstream file(filename);
    if (!file.is_open()) return;

    if (ext == ".csv") {
        file << "\"ID\",\"Tag\"" << std::endl;
        for (const auto& [key, value] : ref_map) {
            file << key << ", " << value << std::endl;
        }
    } else if (ext == ".dat") {
        std::string kind_str = (kind == RefMapKind::Solution) ? "Solution" : "Marker";
        file << "TITLE = \"" << kind_str << " ref map\"" << std::endl;
        file << "VARIABLES = \"ID\",\"Tag\"" << std::endl;
        file << "ZONE T= \"" << kind_str << " tags\"" << std::endl;
        for (const auto& [key, value] : ref_map) {
            file << key << ", " << value << std::endl;
        }
    } else if (ext == ".json") {
        file << "{";
        bool first = true;
        for (const auto& [key, value] : ref_map) {
            if (!first) file << ", ";
            file << '\'' << value << "': " << key;
            first = false;
        }
        file << "}" << std::endl;
    } else {
        throw std::runtime_error("Unsupported file extension for ref map: " + ext);
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