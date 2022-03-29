find_package(rostest REQUIRED)

set(nmpc_ddp_gtest_list
  TestDDPBipedal
  TestDDPVerticalMotion
  TestDDPCentroidalMotion
  )

set(nmpc_ddp_rostest_list
  TestDDPCartPole
  )

foreach(NAME IN LISTS nmpc_ddp_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_ddp)
endforeach()

foreach(NAME IN LISTS nmpc_ddp_rostest_list)
  add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_ddp)
endforeach()
