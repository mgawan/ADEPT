
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(PyADEPT-Examples LANGUAGES C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "adept")
    set(TARGET_BUILD_DIR ${CMAKE_BINARY_DIR})
    if(NOT ADEPT_USE_PYTHON)
        return()
    endif()
else()
    set(TARGET_BUILD_DIR ${CMAKE_BINARY_DIR})
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

function(CONFIGURE_PYTHON_SCRIPT)
    foreach(_TYPE ${ARGN})
        set(FILENAME py_${_TYPE})
        configure_file(
            ${PROJECT_SOURCE_DIR}/py_${_TYPE}.py
            ${TARGET_BUILD_DIR}/examples/py_examples/py_${_TYPE}
            @ONLY)
    endforeach()
endfunction()

configure_python_script(simple_sw asynch_sw asynch_protein multigpu_dna multigpu_protein)