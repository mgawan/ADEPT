#----------------------------------------------------------------------------------------#
#   Add examples
#----------------------------------------------------------------------------------------#

message(STATUS "Adding example: Async Protein...")
add_subdirectory(asynch_protein)

message(STATUS "Adding example: Async DNA...")
add_subdirectory(asynch_sw)

message(STATUS "Adding example: Simple DNA...")
add_subdirectory(simple_sw)

message(STATUS "Adding example: multi GPU DNA ...")
add_subdirectory(multi_gpu)

message(STATUS "Adding example: multi GPU Protein...")
add_subdirectory(multigpu_protein)

message(STATUS "Adding example: multi GPU Protein with scores only...")
add_subdirectory(multigpu_protein_scoreonly)

if (ADEPT_USE_PYTHON)
    message(STATUS "Adding PyADEPT examples...")
    add_subdirectory(py_examples)
endif()