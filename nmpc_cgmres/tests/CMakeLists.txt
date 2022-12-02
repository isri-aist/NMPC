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
  function(catkin_add_gtest NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest)
    gtest_discover_tests(${NAME})
  endfunction()
endif()

foreach(NAME IN LISTS nmpc_cgmres_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} PUBLIC nmpc_cgmres)
endforeach()
