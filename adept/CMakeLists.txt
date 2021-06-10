
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(adept LANGUAGES CXX)

#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
file(GLOB SOURCE_FILES *.cpp)

set(HEADER_FILES "*kernel.hpp driver.hpp")

# enable_language(CUDA)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")

add_library(adept_obj OBJECT ${SOURCE_FILES})
add_library(adept_lib_shared SHARED $<TARGET_OBJECTS:adept_obj>)
set_property(TARGET adept_obj adept_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)

set_property(TARGET adept_obj adept_lib_shared PROPERTY INTERFACE_COMPILE_OPTIONS)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
install(TARGETS adept_lib_shared LIBRARY DESTINATION lib)

