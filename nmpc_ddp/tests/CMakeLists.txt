if(NOT NMPC_STANDALONE)
  find_package(rostest REQUIRED)
endif()

if(NMPC_STANDALONE)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  function(add_nmpc_ddp_test NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest nmpc_ddp)
    gtest_discover_tests(${NAME})
  endfunction()
else()
  function(add_nmpc_ddp_test NAME)
    catkin_add_gtest(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} nmpc_ddp)
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
  add_nmpc_ddp_test(${NAME})
endforeach()

if(NOT NMPC_STANDALONE)
  foreach(NAME IN LISTS nmpc_ddp_rostest_list)
    add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
    target_link_libraries(${NAME} nmpc_ddp)
  endforeach()
endif()
