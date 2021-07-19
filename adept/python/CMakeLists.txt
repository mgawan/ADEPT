cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(pyadept LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)

set(HEADER_FILES "*.hpp")

pybind11_add_module(pyadept SHARED ${SOURCE_FILES})

target_include_directories(pyadept PUBLIC ${PYBIND11_INCLUDE_DIR})

install(TARGETS pyadept LIBRARY DESTINATION lib)