# min cmake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# pyadept
project(pyadept LANGUAGES CXX)

# need C++-17 and position independent code to build bindings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)

set(HEADER_FILES *.hpp)

# add a pybind11 module
pybind11_add_module(adept MODULE ${SOURCE_FILES})

set_target_properties(adept PROPERTIES
    PREFIX                      ""
    SUFFIX                      "${PYTHON_MODULE_EXTENSION}"
    LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    PDB_OUTPUT_DIRECTORY        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    INSTALL_RPATH_USE_LINK_PATH ON
    ${EXTRA_PROPERTIES})

# set include directories
target_include_directories(adept PUBLIC ${PYBIND11_INCLUDE_DIR} ${ADEPT_HEADERS})

# link ADEPT library
target_link_libraries(adept PRIVATE adept_lib_shared)

# install the python lib
install(TARGETS adept LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})