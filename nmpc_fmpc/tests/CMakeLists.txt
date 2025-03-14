if(NOT NMPC_STANDALONE)
  find_package(ament_cmake_gtest REQUIRED)
endif()

if(NMPC_STANDALONE)
  # find_package(ament_cmake_gtest REQUIRED)
  include(GoogleTest)
  find_package(GTest REQUIRED)
  function(add_nmpc_fmpc_test NAME)
    add_executable(${NAME} src/${NAME}.cpp)
    target_link_libraries(${NAME} PUBLIC GTest::gtest nmpc_fmpc)
    gtest_discover_tests(${NAME})
  endfunction()
else()
  function(add_nmpc_fmpc_test NAME)
    ament_add_gtest(${NAME} src/${NAME}.cpp TIMEOUT 200)
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
    ament_add_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp TIMEOUT 200)
    target_link_libraries(${NAME} nmpc_fmpc)
  endforeach()
endif()
