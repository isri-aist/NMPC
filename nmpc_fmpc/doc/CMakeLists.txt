find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
  set(DOXYFILE_PATH ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  add_custom_target(nmpc_fmpc_doc ALL
    ${DOXYGEN_EXECUTABLE} ${DOXYFILE_PATH}
    DEPENDS ${DOXYFILE_PATH}
    COMMENT "Generating Doxygen documentation"
    )
endif()
