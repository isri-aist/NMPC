add_library(NOCDDP
  DDP.cpp
  )
target_link_libraries(NOCDDP PUBLIC
  ${catkin_LIBRARIES}
  )
if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(NOCDDP PUBLIC -march=native)
endif()
