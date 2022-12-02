if(NOT NMPC_STANDALONE)
  find_package(rostest REQUIRED)
endif()

if(NMPC_STANDALONE)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  function(catkin_add_gtest NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest)
    gtest_discover_tests(${NAME})
  endfunction()
endif()


set(nmpc_ddp_gtest_list
  TestBoxQP
  TestDDPBipedal
  TestDDPVerticalMotion
  TestDDPCentroidalMotion
  )

set(nmpc_ddp_rostest_list
  TestDDPCartPole
  )

foreach(NAME IN LISTS nmpc_ddp_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} PUBLIC nmpc_ddp)
endforeach()

if(NOT NMPC_STANDALONE)
  foreach(NAME IN LISTS nmpc_ddp_rostest_list)
    add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC nmpc_ddp)
  endforeach()
endif()
