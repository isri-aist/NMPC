cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME nmpc_fmpc)
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
project(nmpc_fmpc LANGUAGES CXX)

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
else()
  set(CATKIN_ENABLE_TESTING OFF)
  add_project_dependency(nmpc_ddp REQUIRED)
endif()

add_library(nmpc_fmpc INTERFACE)
target_compile_features(nmpc_fmpc INTERFACE cxx_std_17)
if(NOT NMPC_STANDALONE)
target_include_directories(nmpc_fmpc INTERFACE
  ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(nmpc_fmpc INTERFACE
  ${catkin_LIBRARIES}
  )
endif()

target_include_directories(nmpc_fmpc INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
if(TARGET nmpc_ddp::nmpc_ddp)
  target_link_libraries(nmpc_fmpc INTERFACE nmpc_ddp::nmpc_ddp)
endif()

if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_fmpc INTERFACE -march=native)
endif()

install(TARGETS nmpc_fmpc EXPORT "${TARGETS_EXPORT_NAME}")
install(DIRECTORY include/nmpc_fmpc DESTINATION "${INCLUDE_INSTALL_DIR}")

if(BUILD_TESTING OR CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
