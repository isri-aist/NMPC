cmake_minimum_required(VERSION 3.17)

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

project(nmpc_fmpc LANGUAGES CXX)

# Options
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)

if(NOT DEFINED NMPC_STANDALONE)
  set(NMPC_STANDALONE OFF)
endif()

if(NOT NMPC_STANDALONE)
  find_package(nmpc_ddp REQUIRED)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(std_srvs REQUIRED)

  # Eigen
  find_package(Eigen3 REQUIRED)
  include_directories(${EIGEN3_INCLUDE_DIR})

  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt" "")
  endif()

  ament_package()
else()
  set(BUILD_TESTING OFF)
  find_package(nmpc_ddp REQUIRED)
endif()

add_library(nmpc_fmpc INTERFACE)
target_compile_features(nmpc_fmpc INTERFACE cxx_std_17)

target_include_directories(nmpc_fmpc INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
if(TARGET nmpc_ddp::nmpc_ddp)
  target_link_libraries(nmpc_fmpc INTERFACE nmpc_ddp::nmpc_ddp)
endif()

if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_fmpc INTERFACE -march=native)
endif()

install(TARGETS nmpc_fmpc EXPORT "${TARGETS_EXPORT_NAME}")
install(DIRECTORY include/nmpc_fmpc DESTINATION include)

if(NOT NMPC_STANDALONE)
  ament_target_dependencies(nmpc_fmpc INTERFACE
    rclcpp
    std_srvs
    visualization_msgs
  )
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
