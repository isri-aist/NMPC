find_package(rostest REQUIRED)

set(nmpc_fmpc_gtest_list
  TestFmpcOscillator
  )

set(nmpc_fmpc_rostest_list
  TestFmpcCartPole
  )

foreach(NAME IN LISTS nmpc_fmpc_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_fmpc)
endforeach()

foreach(NAME IN LISTS nmpc_fmpc_rostest_list)
  add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_fmpc)
endforeach()
