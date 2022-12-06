# >=3.14 required for CMP0087
cmake_minimum_required(VERSION 3.14)

set(PROJECT_VERSION 0.1.0)
project(NMPC)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Set a default build type to 'RelwithDebInfo' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT ENV{CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

include(CTest)
enable_testing()

# Check if the install prefix is writable by the current user and set SUDO_CMD accordingly
set(SUDO_CMD)
if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX} OUTPUT_QUIET ERROR_QUIET)
endif()
set(TEST_FILE "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}.writable")
# Same as file(TOUCH ...) but ignores failure
execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${TEST_FILE} OUTPUT_QUIET ERROR_QUIET)
if(NOT EXISTS ${TEST_FILE})
  find_program(SUDO_CMD sudo)
  if(NOT SUDO_CMD)
    message(FATAL_ERROR "sudo is not installed on this system and the install prefix (${CMAKE_INSTALL_PREFIX}) is not writable by the current user.
You can try the following solutions:
    - Choose a different installation prefix;
    - Install sudo;
    - Fix the permissions and try again;")
  endif()
  message("-- Use sudo for install: ${SUDO_CMD}")
  set(SUDO_CMD ${SUDO_CMD} -E)
else()
  file(REMOVE ${TEST_FILE})
endif()

include(ExternalProject)

function(add_nmpc_project NAME)
  file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/*")
  ExternalProject_Add(${NAME}
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      -DINSTALL_DOCUMENTATION=${INSTALL_DOCUMENTATION}
      -DBUILD_TESTING=${BUILD_TESTING}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DNMPC_STANDALONE=ON
      INSTALL_COMMAND ${SUDO_CMD}
    TEST_COMMAND ctest -C $<CONFIG>
    INSTALL_COMMAND ${SUDO_CMD} ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/${NAME} --target install --target install --config $<CONFIG>
    USES_TERMINAL_INSTALL TRUE
    STEP_TARGETS configure install
    ${ARGN}
  )
  ExternalProject_Add_Step(${NAME} check-sources
    DEPENDERS configure
    DEPENDS ${SOURCES}
  )
endfunction()
add_nmpc_project(nmpc_cgmres)
add_nmpc_project(nmpc_ddp)
add_nmpc_project(nmpc_fmpc DEPENDS nmpc_ddp)

install(CODE "message(\"Installation done\")")
