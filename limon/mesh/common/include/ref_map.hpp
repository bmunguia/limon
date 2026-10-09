#pragma once

#include <map>
#include <string>

namespace limon {

class RefMap {
public:
    /**
     * Name of a marker or solution field: the map entry, or "REF_<id>" when absent.
     */
    static std::string getRefName(const std::map<int, std::string>& ref_map, int ref_id);
};

} // namespace limon
