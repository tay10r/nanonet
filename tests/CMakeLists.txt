cmake_minimum_required(VERSION 3.14.7)

find_package(Eigen3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_executable(NanoNet_Tests
  ops.cpp
  train.cpp)

target_link_libraries(NanoNet_Tests
  PUBLIC
    GTest::gtest
    GTest::gtest_main
    Eigen3::Eigen
    NanoNet::Compiled)

set_target_properties(NanoNet_Tests
  PROPERTIES
    OUTPUT_NAME run_tests
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_test(NAME NanoNet_UnitTests
  COMMAND $<TARGET_FILE:NanoNet_Tests>)

enable_testing()
