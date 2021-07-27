cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(adept LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
set(default_build_type "Release")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Building Examples" ON)
option(BUILD_TESTS "Building Tests" ON)
##########################################################################################
#       Add subdirectories
##########################################################################################

# Add pyAdept
option(ADEPT_USE_PYTHON "Adding PyAdept" OFF)

if (ADEPT_USE_PYTHON)
    message(STATUS "Adding PyAdept support...")
    add_subdirectory(pyadept)
    message(STATUS "Adding external...")
    add_subdirectory(external)
endif()

include_directories("adept")

message(STATUS "Adding Adept library")

# Add adept library code
add_subdirectory(adept)

# Add examples
if(BUILD_EXAMPLES)
    message(STATUS "Adding Adept examples")
    add_subdirectory(examples)
endif()

# Add tests
if(BUILD_TESTS)
    message(STATUS "Adding ADEPT tests")
    include(CTest)

    add_test(simple_sw ./examples/simple_sw/simple_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)
    add_test(asynch_dna ./examples/asynch_sw/asynch_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)
    add_test(asynch_prot ./examples/asynch_protein/asynch_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_expected256.algn)
    add_test(multi_gpu ./examples/multi_gpu/multi_gpu ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)
endif()