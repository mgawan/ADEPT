
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(adept LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)

set(ADEPT_HEADERS "*kernel.hpp driver.hpp")

# enable_language(CUDA)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")

add_library(adept_obj OBJECT ${SOURCE_FILES})
add_library(adept_lib_shared SHARED $<TARGET_OBJECTS:adept_obj>)

set_property(TARGET adept_obj adept_lib_shared PROPERTY CUDA_ARCHITECTURES 80)

set_property(TARGET adept_obj adept_lib_shared PROPERTY INTERFACE_COMPILE_OPTIONS)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

install(TARGETS adept_lib_shared LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Python bindings
#
if (ADEPT_USE_PYTHON)
    message(STATUS "Adding Adept/python...")
    add_subdirectory(python)
endif()

