cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# SM_80 = A100, SM_90 = H100. PTX JIT handles newer GPUs at runtime.
# Override at cmake time: -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Use pre-cloned cuco if available (avoids needing internet on compute node).
# Pre-clone on login node: git clone --depth 1 --branch v0.0.1
#   https://github.com/NVIDIA/cuCollections.git ~/cpm_cache/cuCollections-v0.0.1
if(DEFINED ENV{CUCO_SOURCE_DIR} AND EXISTS "$ENV{CUCO_SOURCE_DIR}")
    message(STATUS "Using local cuCollections from $ENV{CUCO_SOURCE_DIR}")
    add_subdirectory($ENV{CUCO_SOURCE_DIR} cuco EXCLUDE_FROM_ALL)
else()
    message(STATUS "Downloading CPM and cuCollections from GitHub")
    file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
    )
    include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
    CPMAddPackage(
      NAME cuco
      GITHUB_REPOSITORY NVIDIA/cuCollections
      GIT_TAG main
      OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARKS OFF"
        "BUILD_EXAMPLES OFF"
    )
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gpu_benchmark benchmark.cu)

target_link_libraries(gpu_benchmark
    cuco
    CUDA::cudart
)