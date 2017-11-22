cmake_minimum_required(VERSION 2.8)
# set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_LINK_FLAGS -lm)

project(dolphin_private)

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS -arch=sm_61 -O3)

# Specify binary name and source file to build it from
set(SOURCE_FILES
        lib/cvxgen/ldl.cpp
        lib/cvxgen/matrix_support.cpp
        lib/cvxgen/solver.cpp
        lib/cvxgen/util.cpp
        src/app/main.cpp
        src/finance/asset.cpp
        src/finance/portfolio.cpp
        src/optimization/optimizer.cu
		src/helpers/date.cpp
        src/parsing/parse.cpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

cuda_add_executable(dolphin ${SOURCE_FILES})
