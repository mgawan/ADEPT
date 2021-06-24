cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CXX)
set(CMAKE_CUDA_STANDARD 11)

file(GLOB SOURCE_FILES *.cpp)
set(HEADER_FILES "*kernel.hpp driver.hpp")

set(GPU_TARGETS "gfx908" CACHE STRING "GPU targets to compile for")
find_package(hip)

add_library(adept_lib ${SOURCE_FILES})
target_include_directories(adept_lib PUBLIC ${ROCM_PATH}/include/)

install(TARGETS adept_lib LIBRARY DESTINATION lib)

