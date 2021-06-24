cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(adept)
include_directories("adept")

add_executable(adept_test main.cpp)
target_link_libraries(adept_test adept_lib)
