cmake_minimum_required(VERSION 3.12)
project(pymeshb LANGUAGES C CXX)

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
set(PYMESHB_UTIL_HEADERS
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/util.hpp)
set(PYMESHB_UTIL_SOURCES
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/util.cpp)

# Build libgmf submodule
pybind11_add_module(libgmf MODULE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/gmf/libgmf.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/gmf/src/element.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/gmf/src/mesh.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/gmf/src/solution.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/ref_map.cpp
    ${PYMESHB_UTIL_SOURCES})
target_link_libraries(libgmf PRIVATE Meshb.7)
target_include_directories(libgmf PRIVATE
    ${PROJECT_SOURCE_DIR}/libMeshb/sources
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh)
install(TARGETS libgmf DESTINATION pymeshb/mesh/gmf)

# Build libsu2 submodule
pybind11_add_module(libsu2 MODULE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/libsu2.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/src/element.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/src/mesh.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/src/solution.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/ref_map.cpp
    ${PYMESHB_UTIL_SOURCES})
target_include_directories(libsu2 PRIVATE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh)
install(TARGETS libsu2 DESTINATION pymeshb/mesh/su2)

# Build metric module
pybind11_add_module(_metric MODULE
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/metric.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/decompose.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/recompose.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/integrate.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/normalize.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/perturb.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/edge_length.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/metric/src/rotation_angle.cpp)
target_link_libraries(_metric PRIVATE Eigen3::Eigen)
install(TARGETS _metric DESTINATION pymeshb/metric)
