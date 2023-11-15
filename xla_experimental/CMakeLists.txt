cmake_minimum_required(VERSION 3.10)
project(execute_hlo CXX)

set(PJRT_C_API_LIB xla/bazel-bin/xla/pjrt/c/libpjrt_c_api_cpu.a)
set(XLA_INCLUDE xla)

add_executable(execute_hlo execute_hlo.cpp)
target_link_libraries(execute_hlo "${CMAKE_SOURCE_DIR}/${PJRT_C_API_LIB}")
target_include_directories(execute_hlo PRIVATE "${CMAKE_SOURCE_DIR}/${XLA_INCLUDE}")

set_target_properties(execute_hlo PROPERTIES RUNTIME_OUTPUT_DIRECTORY /build/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_BUILD_TYPE Release)

add_custom_command(TARGET execute_hlo POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:execute_hlo> ${CMAKE_SOURCE_DIR}/build/)