# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and language
project(execute_hlo CXX)

set(XLA_EXTENSION_LIB ./xla_extension/lib)
set(XLA_EXTENSION_INCLUDE ./xla_extension/include)

# Add an executable target with the source file
add_executable(execute_hlo execute_hlo.cpp)

# Link the xla_extension library to the target
target_link_libraries(execute_hlo ${XLA_EXTENSION_LIB})

# Add the xla_extension include directories to the target
target_include_directories(execute_hlo PRIVATE ${XLA_EXTENSION_INCLUDE})

# Set the output directory for the executable object file
set_target_properties(execute_hlo PROPERTIES RUNTIME_OUTPUT_DIRECTORY /build/)