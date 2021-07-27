# min cmake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# pyadept
project(pyadept LANGUAGES CXX)

# need C++-17 and position independent code to build bindings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)

set(HEADER_FILES *.hpp)

# add a pybind11 module
pybind11_add_module(pyadept SHARED ${SOURCE_FILES})

# set include directories
target_include_directories(pyadept PUBLIC ${PYBIND11_INCLUDE_DIR} ${ADEPT_HEADERS})

# link ADEPT library
target_link_libraries(pyadept PRIVATE adept_lib_shared)

# install the python lib
install(TARGETS pyadept LIBRARY DESTINATION lib)