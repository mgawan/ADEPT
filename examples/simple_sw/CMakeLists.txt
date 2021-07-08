
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(simple_sw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simple_sw simple_sw.cpp)
target_link_libraries(simple_sw adept_lib_shared)
