add_library(NOCDDP
  DDP.cpp
  )
target_link_libraries(NOCDDP PUBLIC
  ${catkin_LIBRARIES}
  )
