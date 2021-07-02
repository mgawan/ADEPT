
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(asynch_sw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(asynch_sw asynch_sw.cpp)
target_link_libraries(asynch_sw adept_lib_shared)
