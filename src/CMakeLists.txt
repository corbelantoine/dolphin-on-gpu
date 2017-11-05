# Set compilation flags
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -Weffc++ -pedantic -DDOLPHIN_GPUVERSION=${DOLPHIN_GPU_VERSION}")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-Os")

# Include directories for compilation (-I)
include_directories(
  .
  )

# Get objects libraries from sources
add_subdirectory(app)

# Create the executable with project name
add_executable(DolphinApp
  $<TARGET_OBJECTS:app_objs>
  )
