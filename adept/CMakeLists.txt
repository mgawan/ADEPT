cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set (MY_PLATFORM $ENV{HIP_PLATFORM})
message(STATUS "HIP_PLATFORM is set to: ${MY_PLATFORM}")

if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
        message(STATUS "Building for CUDA devices")
        project(adept LANGUAGES CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    endif()
else()
     message(STATUS "Building for AMD devices")
    find_package(hip)
endif()

#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)
set(HEADER_FILES "*kernel.hpp driver.hpp")

if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
        enable_language(CUDA)
        set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
    endif()
endif()


add_library(adept_obj OBJECT ${SOURCE_FILES})
add_library(adept_lib_shared SHARED $<TARGET_OBJECTS:adept_obj>)

if(MY_PLATFORM STREQUAL "amd")
    target_compile_definitions(adept_obj PUBLIC PLATFORM_AMD=true)
endif()

if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
        set_property(TARGET adept_obj adept_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)
    endif()
endif()


get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
target_link_libraries(adept_lib_shared hip::device)
install(TARGETS adept_lib_shared LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Python bindings
#
#if (ADEPT_USE_PYTHON)
 #   message(STATUS "Adding Adept/python...")
 #   add_subdirectory(python)
#endif()