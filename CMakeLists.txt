cmake_minimum_required(VERSION 3.8)
project(dolphin_private)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        lib/json.hpp
        src/app/main.cpp
        src/finance/asset.cpp
        src/finance/asset.hpp
        src/finance/portfolio.cpp
        src/finance/portfolio.hpp
        src/optimization/optimizer.cpp
        src/optimization/optimizer.hpp
		    src/helpers/date.cpp
		    src/helpers/date.hpp
        src/parsing/parse.cpp
        src/parsing/parse.hpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

add_executable(dolphin ${SOURCE_FILES})
