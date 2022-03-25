find_package(rostest REQUIRED)

set(NOCDDP_gtest_list
  TestDDPBipedal
  TestDDPVerticalMotion
  )

set(NOCDDP_rostest_list
  TestDDPCartPole
  )

foreach(NAME IN LISTS NOCDDP_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} NOCDDP)
endforeach()

foreach(NAME IN LISTS NOCDDP_rostest_list)
  add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
  target_link_libraries(${NAME} NOCDDP)
endforeach()
