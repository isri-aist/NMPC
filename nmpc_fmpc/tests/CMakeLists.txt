set(nmpc_fmpc_gtest_list
  TestFmpcBipedal
  )

foreach(NAME IN LISTS nmpc_fmpc_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} nmpc_fmpc)
endforeach()
