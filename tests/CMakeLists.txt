set(NOCDDP_gtest_list
  TestDDPBipedal
  TestDDPCartPole
  )

foreach(NAME IN LISTS NOCDDP_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} NOCDDP)
endforeach()
