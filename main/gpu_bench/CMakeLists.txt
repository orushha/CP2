cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# fetch cuCollections — header only, no install needed
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
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