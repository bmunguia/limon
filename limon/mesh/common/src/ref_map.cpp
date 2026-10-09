#include "../include/ref_map.hpp"

namespace limon {

std::string RefMap::getRefName(const std::map<int, std::string>& ref_map, int ref_id) {
    auto it = ref_map.find(ref_id);
    if (it != ref_map.end()) {
        return it->second;
    }
    return "REF_" + std::to_string(ref_id);
}

} // namespace limon
