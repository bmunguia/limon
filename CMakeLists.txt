cmake_minimum_required(VERSION 3.12)
project(limon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Look in your conda environment for packages
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

# Required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# libMeshb submodule
add_subdirectory(${PROJECT_SOURCE_DIR}/libMeshb)

# Shared util files
set(LIMON_UTIL_HEADERS
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/include/util.hpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/include/ref_map.hpp)
set(LIMON_UTIL_SOURCES
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/src/util.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/src/ref_map.cpp)

# Build libgmf submodule
pybind11_add_module(libgmf MODULE
    ${PROJECT_SOURCE_DIR}/limon/mesh/gmf/libgmf.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/gmf/src/element.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/gmf/src/mesh.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/gmf/src/solution.cpp
    ${LIMON_UTIL_SOURCES})
target_link_libraries(libgmf PRIVATE Meshb.7)
target_include_directories(libgmf PRIVATE
    ${PROJECT_SOURCE_DIR}/libMeshb/sources
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/include)
install(TARGETS libgmf DESTINATION limon/mesh/gmf)

# Build libsu2 submodule
pybind11_add_module(libsu2 MODULE
    ${PROJECT_SOURCE_DIR}/limon/mesh/su2/libsu2.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/su2/src/element.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/su2/src/mesh.cpp
    ${PROJECT_SOURCE_DIR}/limon/mesh/su2/src/solution.cpp
    ${LIMON_UTIL_SOURCES})
target_include_directories(libsu2 PRIVATE
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/include)
install(TARGETS libsu2 DESTINATION limon/mesh/su2)

# Build metric module
pybind11_add_module(_metric MODULE
    ${PROJECT_SOURCE_DIR}/limon/numerics/metric.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/decompose.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/recompose.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/integrate.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/normalize.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/perturb.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/edge_length.cpp
    ${PROJECT_SOURCE_DIR}/limon/numerics/src/rotation_angle.cpp)
target_link_libraries(_metric PRIVATE Eigen3::Eigen)
install(TARGETS _metric DESTINATION limon/numerics)

# Build ref_map module
pybind11_add_module(_ref_map MODULE
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/src/ref_map_bindings.cpp
    ${LIMON_UTIL_SOURCES})
target_include_directories(_ref_map PRIVATE
    ${PROJECT_SOURCE_DIR}/limon/mesh/common/include)
install(TARGETS _ref_map DESTINATION limon/mesh)

# Build refine module
pybind11_add_module(_refine MODULE
    ${PROJECT_SOURCE_DIR}/limon/operations/src/refine.cpp
    ${PROJECT_SOURCE_DIR}/limon/operations/src/refine_2d.cpp
    ${PROJECT_SOURCE_DIR}/limon/operations/src/refine_3d.cpp)
install(TARGETS _refine DESTINATION limon/operations)