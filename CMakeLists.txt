
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(adept_test LANGUAGES CXX)

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

message(STATUS "The GPU selected is: ${ADEPT_GPU}")

if (ADEPT_GPU STREQUAL "NVIDIA")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version")

elseif (ADEPT_GPU STREQUAL "AMD")
    # todo
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

# C++ standard = 17 needed
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "")

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
