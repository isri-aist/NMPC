cmake_minimum_required(VERSION 3.1)
project(nmpc_fmpc)

add_compile_options(-std=c++17)

# Options
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)

find_package(catkin REQUIRED COMPONENTS
  nmpc_ddp
  )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS
  nmpc_ddp
  DEPENDS EIGEN3
  INCLUDE_DIRS include
  LIBRARIES
  )

add_library(nmpc_fmpc INTERFACE)
target_include_directories(nmpc_fmpc INTERFACE
  include
  ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(nmpc_fmpc INTERFACE
  ${catkin_LIBRARIES}
  )
if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_fmpc INTERFACE -march=native)
endif()

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
