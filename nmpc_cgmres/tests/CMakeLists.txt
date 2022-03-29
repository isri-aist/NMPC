find_package(rostest REQUIRED)

set(nmpc_cgmres_gtest_list
  TestGmres
  TestCgmresSolver
  )

foreach(NAME IN LISTS nmpc_cgmres_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_cgmres)
endforeach()
