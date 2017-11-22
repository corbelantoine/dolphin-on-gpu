cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_LINK_FLAGS "-lm")

project(dolphin_private)

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "-x cu")

set(SOURCE_FILES
        lib/cvxgen/ldl.cpp
        lib/cvxgen/matrix_support.cpp
        lib/cvxgen/solver.cpp
        lib/cvxgen/util.cpp
        src/app/main.cpp
        src/finance/asset.cpp
        src/finance/portfolio.cpp
  #      src/optimization/optimizer.cpp
		src/helpers/date.cpp
        src/parsing/parse.cpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

# Specify binary name and source file to build it from
cuda_add_executable(
    optimizer
    src/optimization/optimizer.cpp)

add_executable(dolphin ${SOURCE_FILES})
