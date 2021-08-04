cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# find_package(Threads::Threads REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
file(GLOB SOURCE_FILES *.cpp)
set(HEADER_FILES "*kernel.hpp driver.hpp")
enable_language(CUDA)
set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(adept_obj OBJECT ${SOURCE_FILES})
add_library(adept_lib_shared SHARED $<TARGET_OBJECTS:adept_obj>)
set_property(TARGET adept_obj adept_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)

# set_property(TARGET adept_obj adept_lib_shared PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
# target_link_libraries(adept_lib_shared PUBLIC OpenMP::OpenMP_CXX)
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