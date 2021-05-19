cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
file(GLOB SOURCE_FILES *.cpp)
set(HEADER_FILES "*kernel.hpp driver.hpp")
enable_language(CUDA)
set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

add_library(adept_obj OBJECT ${SOURCE_FILES})
add_library(adept_lib_shared SHARED $<TARGET_OBJECTS:adept_obj>)
set_property(TARGET adept_obj adept_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)
install(TARGETS adept_lib_shared LIBRARY DESTINATION lib)

