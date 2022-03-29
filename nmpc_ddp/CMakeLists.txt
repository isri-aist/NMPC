cmake_minimum_required(VERSION 3.1)
project(nmpc_ddp)

add_compile_options(-std=c++17)

# Options
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS
  roscpp
  DEPENDS EIGEN3
  INCLUDE_DIRS include
  LIBRARIES nmpc_ddp
  )

add_library(nmpc_ddp INTERFACE)
target_include_directories(nmpc_ddp INTERFACE
  include
  ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(nmpc_ddp INTERFACE
  ${catkin_LIBRARIES}
  )
if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_ddp INTERFACE -march=native)
endif()

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()
