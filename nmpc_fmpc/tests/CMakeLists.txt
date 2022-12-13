if(NOT NMPC_STANDALONE)
  find_package(rostest REQUIRED)
endif()

if(NMPC_STANDALONE)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  function(add_nmpc_fmpc_test NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest nmpc_fmpc)
    gtest_discover_tests(${NAME})
  endfunction()
else()
  function(add_nmpc_fmpc_test NAME)
    catkin_add_gtest(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} nmpc_fmpc)
  endfunction()
endif()



set(nmpc_fmpc_gtest_list
  TestMathUtils
  TestFmpcOscillator
  )

set(nmpc_fmpc_rostest_list
  TestFmpcCartPole
  )

foreach(NAME IN LISTS nmpc_fmpc_gtest_list)
  add_nmpc_fmpc_test(${NAME})
endforeach()

if(NOT NMPC_STANDALONE)
  foreach(NAME IN LISTS nmpc_fmpc_rostest_list)
    add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
    target_link_libraries(${NAME} nmpc_fmpc)
  endforeach()
endif()
