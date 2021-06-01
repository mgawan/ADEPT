
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(adept_test LANGUAGES CXX)

# set the CXX compiler to DPC++
# set(CMAKE_CXX_COMPILER dpcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(adept)
include_directories("adept")

add_executable(adept_test main.cpp)
target_link_libraries(adept_test adept_lib_shared)

# install the test script
if(NOT BASH_EXECUTABLE)
    set(BASH_EXECUTABLE "/bin/bash")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_adept.sh
    ${CMAKE_CURRENT_BINARY_DIR}/test_adept @ONLY)