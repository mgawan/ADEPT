cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
set(default_build_type "Release")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Building Examples" ON)
##########################################################################################
#       Add subdirectories
##########################################################################################

include_directories("adept")

message(STATUS "Adding Adept library")

# Add adept library code
add_subdirectory(adept)

# Add examples
if(BUILD_EXAMPLES)
    message(STATUS "Adding Adept examples")
    add_subdirectory(examples)
endif()
##########################################################################################
#       Configure Testing script
##########################################################################################

# install the test script
if(NOT BASH_EXECUTABLE)
    set(BASH_EXECUTABLE "/bin/bash")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_adept.sh
    ${CMAKE_CURRENT_BINARY_DIR}/test_adept @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/run_datasets.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_datasets @ONLY)