
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(adept_test LANGUAGES CXX)

# set the CXX compiler to DPC++
# set(CMAKE_CXX_COMPILER dpcpp)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend '--cuda-gpu-arch=sm_70' -Wno-unknown-cuda-version")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend '--cuda-gpu-arch=sm_70' -Wno-unknown-cuda-version")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(adept)
include_directories("adept")

add_executable(adept_test main.cpp)
target_link_libraries(adept_test adept_lib_shared)


