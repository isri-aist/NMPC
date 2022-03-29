add_library(nmpc_ddp
  DDP.cpp
  )
target_link_libraries(nmpc_ddp PUBLIC
  ${catkin_LIBRARIES}
  )
if(OPTIMIZE_FOR_NATIVE)
  target_compile_options(nmpc_ddp PUBLIC -march=native)
endif()
