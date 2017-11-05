cmake_minimum_required(VERSION 3.0.2)
project(Dolphin_GPU CXX)
set(DOLPHIN_GPU_VERSION 0.1)

# direcory variables
set(SRC_DIR src/)
# set(DOC_DIR doc/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

# source directories
add_subdirectory(${SRC_DIR})
# add_subdirectory(${DOC_DIR})
