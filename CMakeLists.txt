cmake_minimum_required(VERSION 4.0)
project(pymeshb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# Add the libMeshb subdirectory
add_subdirectory(libMeshb/sources)
target_compile_options(Meshb.7 PRIVATE -fPIC)

# Add the pymeshb Python module
pybind11_add_module(libmeshb pymeshb/libMeshb/libmeshb.cpp)

# Link the Meshb.7 library from libMeshb to the pymeshb module
target_link_libraries(libmeshb PRIVATE Meshb.7)
target_include_directories(pymeshb PRIVATE libMeshb/sources)