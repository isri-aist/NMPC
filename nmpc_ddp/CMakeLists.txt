cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME nmpc_ddp)
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
project(nmpc_ddp LANGUAGES CXX)

# Options
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)

if(NOT DEFINED NMPC_STANDALONE)
  set(NMPC_STANDALONE OFF)
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
    LIBRARIES
    )
else()
  set(CATKIN_ENABLE_TESTING OFF)
endif()

add_library(nmpc_ddp INTERFACE)
if(NOT NMPC_STANDALONE)
  target_include_directories(nmpc_ddp INTERFACE
    include
    ${catkin_INCLUDE_DIRS}
    )
  target_link_libraries(nmpc_ddp INTERFACE
    ${catkin_LIBRARIES}
    )
else()
  target_include_directories(nmpc_ddp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
endif()
if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_ddp INTERFACE -march=native)
endif()
if(TARGET Eigen3::Eigen)
  target_link_libraries(nmpc_ddp INTERFACE Eigen3::Eigen)
else()
  target_include_directories(nmpc_ddp SYSTEM INTERFACE "${EIGEN3_INCLUDE_DIR}")
endif()

install(TARGETS nmpc_ddp EXPORT "${TARGETS_EXPORT_NAME}")
install(DIRECTORY include/nmpc_ddp DESTINATION "${INCLUDE_INSTALL_DIR}")

if(BUILD_TESTING OR CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
