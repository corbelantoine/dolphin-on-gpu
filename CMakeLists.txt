cmake_minimum_required(VERSION 3.8)
project(dolphin_private)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        lib/json.hpp
        src/app/main.cc
        src/finance/asset.cc
        src/finance/asset.hh
        src/finance/portfolio.cc
        src/finance/portfolio.hh
        src/optimization/optimizer.cc
        src/optimization/optimizer.hh
        src/parsing/parse.cc
        src/parsing/parse.hh)

add_executable(dolphin_private ${SOURCE_FILES})