
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(multi_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multi_gpu multi_gpu.cpp)
target_link_libraries(multi_gpu adept_lib_shared)
