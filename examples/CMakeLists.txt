#----------------------------------------------------------------------------------------#
#   Add examples
#----------------------------------------------------------------------------------------#

message(STATUS "Adding example: Async Protein...")
add_subdirectory(asynch_protein)

message(STATUS "Adding example: multi GPU DNA ...")
add_subdirectory(multi_gpu)

message(STATUS "Adding example: Async DNA...")
add_subdirectory(asynch_sw)

message(STATUS "Adding example: Simple DNA...")
add_subdirectory(simple_sw)
