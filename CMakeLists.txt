##########################################################################################
#       CMake settings
##########################################################################################

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CXX)

# C++ standard = 17 needed
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "")

# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option (ADEPT_INSTR "Enable Instrumentation in ADEPT" ON)

if (ADEPT_INSTR)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DADEPT_INSTR")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DADEPT_INSTR")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -DADEPT_INSTR")
endif()

##########################################################################################
#       GPU selection
##########################################################################################

# defalt GPU type
set(default_gpu "NVIDIA")

# list of supported GPUs
set(GPUS "INTEL" "NVIDIA")

if (NOT ADEPT_GPU)
    message(STATUS "Setting GPU type to '${default_gpu}' as GPU not specified.")
    set(ADEPT_GPU ${default_gpu} CACHE
      STRING "Choose the GPU type" FORCE)

    # Set the possible values of GPUs
    set_property(CACHE ADEPT_GPU PROPERTY STRINGS
        "INTEL" "NVIDIA")
else()
    message(STATUS "The selected GPU type is ${ADEPT_GPU}.")
endif()

# check if correct GPU type chosen
if (NOT ADEPT_GPU IN_LIST GPUS)
    message(FATAL_ERROR "Error! ADEPT_GPU must be set to either INTEL or NVIDIA")
endif()

# enable GPU specific DPC++ flags
if (ADEPT_GPU STREQUAL "NVIDIA")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version -DNVIDIA_GPU")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Xsycl-target-backend --cuda-gpu-arch=sm_70 -Wno-unknown-cuda-version -DNVIDIA_GPU")

elseif (ADEPT_GPU STREQUAL "INTEL")
    # add a flag
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DINTEL_GPU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINTEL_GPU")

elseif (ADEPT_GPU STREQUAL "AMD")
    # do nothing
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAMD_GPU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMD_GPU")
endif()

##########################################################################################
#       Add sub-directories
##########################################################################################

# Add pyAdept
option(ADEPT_USE_PYTHON "Adding PyAdept" OFF)

if (ADEPT_USE_PYTHON)
    message(STATUS "Adding PyAdept support...")
    add_subdirectory(pyadept)
    message(STATUS "Adding external...")
    add_subdirectory(external)
endif()

# include Adept library
include_directories("adept")

# Add adept library code
message(STATUS "Adding Adept library...")
add_subdirectory(adept)


# Add examples
option(BUILD_EXAMPLES "Building Examples" ON)

if(BUILD_EXAMPLES)
    message(STATUS "Adding Adept examples...")
    add_subdirectory(examples)
endif()

# Add CTests
option(BUILD_TESTS "Building Tests" ON)

if(BUILD_TESTS)
    message(STATUS "Adding ADEPT tests...")
    include(CTest)

    # we need examples for testing
    if (NOT BUILD_EXAMPLES)
        set (BUILD_EXAMPLES ON)
        add_subdirectory(examples)
    endif()

    # ADEPT tests

    add_test(simple_dna ./examples/simple_sw/simple_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)

    add_test(async_dna ./examples/asynch_sw/asynch_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)

    add_test(async_protein ./examples/asynch_protein/asynch_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_expected256.algn)

    add_test(multiGPU_dna ./examples/multi_gpu/multi_gpu ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)

    add_test(multiGPU_aa ./examples/multigpu_protein/multigpu_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_expected256.algn)

    # PyADEPT testing

    if (ADEPT_USE_PYTHON)

        add_test(py_simple_dna python ./examples/py_examples/py_simple_sw -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_async_dna python ./examples/py_examples/py_asynch_sw -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_multiGPU_dna python ./examples/py_examples/py_multigpu_dna -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_async_protein python ./examples/py_examples/py_asynch_protein -r ../test-data/protein-reference.fasta -q ../test-data/protein-query.fasta -t ../test-data/protein_expected256.algn)

        add_test(py_multiGPU_aa python ./examples/py_examples/py_multigpu_protein -r ../test-data/protein-reference.fasta -q ../test-data/protein-query.fasta -t ../test-data/protein_expected256.algn)

    endif()

endif()

##########################################################################################
#       Configure testing scripts
##########################################################################################

# install the bash scripts
if(NOT BASH_EXECUTABLE)
    set(BASH_EXECUTABLE "/bin/bash")
endif()

# configure_file(${CMAKE_CURRENT_LIST_DIR}/test_adept.sh
#     ${CMAKE_CURRENT_BINARY_DIR}/test_adept @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/run_datasets.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_datasets @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/roofline_intel.sh
    ${CMAKE_CURRENT_BINARY_DIR}/roofline_intel @ONLY)