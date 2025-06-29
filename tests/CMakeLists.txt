set(QpSolverCollection_gtest_list
  TestQpSolversEnabled
  TestSampleQP
  )

foreach(NAME IN LISTS QpSolverCollection_gtest_list)
  if(DEFINED CATKIN_DEVEL_PREFIX)
    catkin_add_gtest(${NAME} ${NAME}.cpp)
  else()
    find_package(GTest REQUIRED)
    include(GoogleTest)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} GTest::gtest)
    gtest_discover_tests(${NAME})
  endif()
  if(SKIP_PRIVATE_SOLVER_TEST)
    target_compile_definitions(${NAME} PRIVATE SKIP_PRIVATE_SOLVER_TEST)
  endif()
  if(FORCE_ALL_SOLVER_TEST)
    target_compile_definitions(${NAME} PRIVATE FORCE_ALL_SOLVER_TEST)
  endif()
  target_link_libraries(${NAME} QpSolverCollection)
endforeach()

add_executable(SampleQP SampleQP.cpp)
target_link_libraries(SampleQP QpSolverCollection)

add_executable(SampleMPC SampleMPC.cpp)
target_link_libraries(SampleMPC QpSolverCollection)


add_executable(SampleNMPC nmpc.cpp)
target_link_libraries(SampleNMPC QpSolverCollection)

add_executable(SampleNMPC2 nmpc2.cpp)
target_link_libraries(SampleNMPC2 QpSolverCollection)

add_executable(SampleDebug debug.cpp)
target_link_libraries(SampleDebug QpSolverCollection)

add_executable(SampleMPC2 SampleMPC2.cpp)
target_link_libraries(SampleMPC2 QpSolverCollection)

add_executable(Samplesqp sample_sqp.cpp)
target_link_libraries(Samplesqp QpSolverCollection)

add_executable(Samplesqp2 sample_sqp2.cpp)
target_link_libraries(Samplesqp2 QpSolverCollection)

add_executable(Samplesqp3 sample_sqp3.cpp)
target_link_libraries(Samplesqp3 QpSolverCollection)

add_executable(ipopt_test ipopt_test.cpp)
target_link_libraries(ipopt_test QpSolverCollection ${Ipopt_LIBRARIES})
