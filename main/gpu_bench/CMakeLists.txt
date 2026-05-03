cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# downloading CPM - package manager for cmake
file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# fetch cuCollections from github
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG v0.0.1
  OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCHMARKS OFF"
    "BUILD_EXAMPLES OFF"
)

find_package(CUDAToolkit REQUIRED)

add_executable(gpu_benchmark benchmark.cu)

target_link_libraries(gpu_benchmark
    cuco
    CUDA::cudart
)