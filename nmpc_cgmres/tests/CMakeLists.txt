find_package(GTest REQUIRED)
include(GoogleTest)

set(nmpc_cgmres_gtest_list
  TestGmres
  TestCgmresSolver
)

function(add_nmpc_cgmres_test NAME)
  add_executable(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} PUBLIC GTest::gtest nmpc_cgmres)
  gtest_discover_tests(${NAME})
endfunction()

foreach(NAME IN LISTS nmpc_cgmres_gtest_list)
  add_nmpc_cgmres_test(${NAME})
endforeach()
