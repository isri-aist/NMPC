cmake_minimum_required(VERSION 3.1)
project(nmpc_cgmres)

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
  LIBRARIES nmpc_cgmres
  )

add_library(nmpc_cgmres
  src/CgmresSolver.cpp
  )
target_include_directories(nmpc_cgmres PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(nmpc_cgmres PUBLIC
  ${catkin_LIBRARIES}
  )
if(OPTIMIZE_FOR_NATIVE)
  # TODO fix segmentation faults
  message(WARNING The -march=native option is disabled for nmpc_cgmres because it causes segmentation faults in tests.)
  # target_compile_options(nmpc_cgmres INTERFACE -march=native)
endif()

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
