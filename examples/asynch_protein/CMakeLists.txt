
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(asynch_sw LANGUAGES CXX)
find_package(hip)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(asynch_protein asynch_protein.cpp)
target_link_libraries(asynch_protein adept_lib_shared hip::device)
