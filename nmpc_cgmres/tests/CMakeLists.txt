if(NOT NMPC_STANDALONE)
  find_package(rostest REQUIRED)
endif()

set(nmpc_cgmres_gtest_list
  TestGmres
  TestCgmresSolver
)

if(NMPC_STANDALONE)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  function(add_nmpc_cgmres_test NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest nmpc_cgmres)
    gtest_discover_tests(${NAME})
  endfunction()
else()
  function(add_nmpc_cgmres_test NAME)
    catkin_add_gtest(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} nmpc_cgmres)
  endfunction()
endif()

foreach(NAME IN LISTS nmpc_cgmres_gtest_list)
  add_nmpc_cgmres_test(${NAME})
endforeach()
