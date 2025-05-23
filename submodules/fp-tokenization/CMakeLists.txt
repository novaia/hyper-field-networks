cmake_minimum_required(VERSION 3.23)
project(fp_tokenization LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "all")

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
    cuda_ffi
    ${CMAKE_CURRENT_LIST_DIR}/csrc/tokenization.cu
    ${CMAKE_CURRENT_LIST_DIR}/csrc/ffi.cpp
)
target_link_libraries(cuda_ffi PRIVATE fmt::fmt)

install(TARGETS cuda_ffi DESTINATION fp_tokenization)
