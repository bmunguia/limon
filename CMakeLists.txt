cmake_minimum_required(VERSION 4.0)
project(pymeshb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the libMeshb submodule
add_subdirectory(${PROJECT_SOURCE_DIR}/libMeshb)
target_compile_options(Meshb.7 PRIVATE -fPIC)

# Add the pymeshb Python module
find_package(pybind11 REQUIRED)
pybind11_add_module(libmeshb MODULE ${PROJECT_SOURCE_DIR}/pymeshb/gamma/libmeshb.cpp)

# Link the Meshb.7 library from libMeshb to the pymeshb module
target_link_libraries(libmeshb PRIVATE Meshb.7)
target_include_directories(libmeshb PRIVATE ${PROJECT_SOURCE_DIR}/libMeshb/sources)