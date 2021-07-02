##########################################################################################
#       CMake settings
##########################################################################################

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CXX)

# C++ standard = 17 needed
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "")

# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option (ADEPT_INSTR "Enable Instrumentation in ADEPT" ON)

if (ADEPT_INSTR)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DADEPT_INSTR")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DADEPT_INSTR")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -DADEPT_INSTR")
endif()

##########################################################################################
#       GPU selection
##########################################################################################

set(default_gpu "Intel")

if (NOT ADEPT_GPU)
    message(STATUS "Setting GPU type to '${default_gpu}' as GPU not specified.")
    set(ADEPT_GPU ${default_gpu} CACHE
      STRING "Choose the GPU type" FORCE)

    # Set the possible values of GPUs
    set_property(CACHE ADEPT_GPU PROPERTY STRINGS
        "Intel" "NVIDIA")
else()
    message(STATUS "The selected GPU type is ${ADEPT_GPU}.")
endif()

if (ADEPT_GPU STREQUAL "NVIDIA")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version")

elseif (ADEPT_GPU STREQUAL "AMD")
    # todo
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

##########################################################################################
#       Add sub-directories
##########################################################################################

include_directories("adept")

# Add adept library code
message(STATUS "Adding Adept library")
add_subdirectory(adept)

# Add examples
message(STATUS "Adding Adept examples")
add_subdirectory(examples)

##########################################################################################
#       Configure testing scripts
##########################################################################################

# install the test script
if(NOT BASH_EXECUTABLE)
    set(BASH_EXECUTABLE "/bin/bash")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_adept.sh
    ${CMAKE_CURRENT_BINARY_DIR}/test_adept @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/run_datasets.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_datasets @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/roofline_intel.sh
    ${CMAKE_CURRENT_BINARY_DIR}/roofline_intel @ONLY)