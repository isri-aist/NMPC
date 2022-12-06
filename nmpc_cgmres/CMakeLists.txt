cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME nmpc_cgmres)
set(PROJECT_GENERATED_HEADERS_SKIP_DEPRECATED ON)
set(PROJECT_GENERATED_HEADERS_SKIP_CONFIG ON)
set(PROJECT_GENERATED_HEADERS_SKIP_WARNING ON)
set(PROJECT_URL https://github.com/isri-aist/NMPC)
set(PROJECT_DESCRIPTION "")
set(CMAKE_CXX_STANDARD 17)
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CXX_DISABLE_WERROR ON)
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

include(../cmake/base.cmake)
project(nmpc_cgmres LANGUAGES CXX)

# Options
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)

if(NOT DEFINED NMPC_STANDALONE)
  set(NMPC_STANDALONE OFF)
  set(DOXYGEN_HTML_OUTPUT html)
else()
  set(DOXYGEN_HTML_OUTPUT doxygen-html)
endif()

if(NOT NMPC_STANDALONE)
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    )
endif()

# Eigen
find_package(Eigen3 QUIET NO_CMAKE_PACKAGE_REGISTRY)
if(Eigen3_FOUND)
  add_project_dependency(Eigen3 REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
else()
  add_project_dependency(Eigen3 MODULE REQUIRED)
endif()

if(NOT NMPC_STANDALONE)
  catkin_package(
    CATKIN_DEPENDS
    roscpp
    DEPENDS EIGEN3
    INCLUDE_DIRS include
    LIBRARIES nmpc_cgmres
    )
else()
  set(CATKIN_ENABLE_TESTING OFF)
endif()

add_library(nmpc_cgmres
  src/CgmresSolver.cpp
  )
target_compile_features(nmpc_cgmres PUBLIC cxx_std_17)
if(NOT NMPC_STANDALONE)
  target_include_directories(nmpc_cgmres PUBLIC
    ${catkin_INCLUDE_DIRS}
    )
  target_link_libraries(nmpc_cgmres PUBLIC
    ${catkin_LIBRARIES}
    )
endif()
target_include_directories(nmpc_cgmres PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

if(TARGET Eigen3::Eigen)
  target_link_libraries(nmpc_cgmres PUBLIC Eigen3::Eigen)
else()
  target_include_directories(nmpc_cgmres SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
endif()

install(TARGETS nmpc_cgmres EXPORT "${TARGETS_EXPORT_NAME}")
install(DIRECTORY include/nmpc_cgmres DESTINATION "${INCLUDE_INSTALL_DIR}")


if(OPTIMIZE_FOR_NATIVE)
  # TODO fix segmentation faults
  message(WARNING "The -march=native option is disabled for nmpc_cgmres because it causes segmentation faults in tests.")
  # target_compile_options(nmpc_cgmres INTERFACE -march=native)
endif()

if(BUILD_TESTING OR CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
