cmake_minimum_required(VERSION 3.23)
project(opengl_proc_gen)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    cpp_ffi
    ${CMAKE_CURRENT_LIST_DIR}/csrc/ffi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csrc/vector_matrix_math.c
)
install(TARGETS cpp_ffi DESTINATION opengl_proc_gen)
