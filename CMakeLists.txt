cmake_minimum_required(VERSION 3.14.7)

project(NanoNet
  DESCRIPTION "A C99 library for executing and training neural networks."
  VERSION 1.0)

option(NANONET_COMPILED "Whether to build the compiled version of the library." ON)
option(NANONET_TESTS    "Whether to build the unit tests." OFF)
option(NANONET_EXAMPLES "Whether to build the example programs." OFF)
option(NANONET_WARNINGS "Whether to build with compiler warnings." OFF)
option(NANONET_DOCS     "Whether to build the documentation." OFF)

add_library(NanoNet INTERFACE)
target_sources(NanoNet INTERFACE include/NanoNet.h)
target_include_directories(NanoNet INTERFACE include)
if(UNIX)
  target_link_libraries(NanoNet INTERFACE m)
endif()
add_library(NanoNet::HeaderOnly ALIAS NanoNet)

if(NANONET_COMPILED)
  add_library(NanoNetC src/NanoNet.c)
  target_include_directories(NanoNetC PUBLIC include)
  target_link_libraries(NanoNetC PUBLIC NanoNet::HeaderOnly)
  add_library(NanoNet::Compiled ALIAS NanoNetC)
  if(NANONET_WARNINGS)
    if(CMAKE_COMPILER_IS_GNUCC)
      target_compile_options(NanoNetC PRIVATE -Wall -Wextra -Werror -Wfatal-errors -fanalyzer)
    endif()
  endif()
endif()

if(NANONET_EXAMPLES)

  function(add_example name)
    set(target NanoNet_Example_${name})
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PUBLIC NanoNet::Compiled)
    set_target_properties(${target}
      PROPERTIES
        OUTPUT_NAME ${name})
  endfunction()

  add_example(custom_allocator examples/custom_allocator.c)
  add_example(status_string examples/status.c)
  add_example(inference examples/inference.c)

endif()

if(NANONET_TESTS)
  add_subdirectory(tests)
  enable_testing()
endif()

if(NANONET_DOCS)
  find_package(Doxygen REQUIRED)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY)
  add_custom_target(NanoNet_Docs ALL
    "${DOXYGEN_EXECUTABLE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
