cmake_minimum_required(VERSION 3.23)
project(fp_conversion LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    cpp_ffi
    ${CMAKE_CURRENT_LIST_DIR}/csrc/ffi.cpp
)

install(TARGETS cpp_ffi DESTINATION fp_conversion)
