cmake_minimum_required(VERSION 3.1)
project(noc_ddp)

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
  LIBRARIES NOCDDP
  )

include_directories(include ${catkin_INCLUDE_DIRS})

add_subdirectory(src)

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()
