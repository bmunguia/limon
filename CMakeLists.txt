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

# Build libmeshb submodule
pybind11_add_module(libmeshb MODULE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/gmf/libmeshb.cpp
    ${PYMESHB_UTIL_SOURCES})
target_link_libraries(libmeshb PRIVATE Meshb.7)
target_include_directories(libmeshb PRIVATE
    ${PROJECT_SOURCE_DIR}/libMeshb/sources
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh)

# Build libsu2 submodule
pybind11_add_module(libsu2 MODULE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/libsu2.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/mesh.cpp
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh/su2/solution.cpp
    ${PYMESHB_UTIL_SOURCES})

target_include_directories(libsu2 PRIVATE
    ${PROJECT_SOURCE_DIR}/pymeshb/mesh)

# Build metric module
pybind11_add_module(_metric MODULE ${PROJECT_SOURCE_DIR}/pymeshb/metric/metric.cpp)
target_link_libraries(_metric PRIVATE Eigen3::Eigen)
