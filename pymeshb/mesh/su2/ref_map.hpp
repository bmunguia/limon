#pragma once

#include <string>
#include <map>
#include <fstream>
#include <stdexcept>

namespace pymeshb {
namespace su2 {

class RefMap {
public:
    static std::map<int, std::string> loadRefMap(const std::string& filename);

    static void saveRefMap(const std::map<int, std::string>& ref_map, const std::string& filename);

    static std::string getRefName(const std::map<int, std::string>& ref_map, int marker_id);
};

} // namespace su2
} // namespace pymeshb