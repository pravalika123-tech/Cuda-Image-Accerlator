cmake_minimum_required(VERSION 3.18)
project(CUDAImageAccelerator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_library(image_accel STATIC
    src/cpu_reference.cpp
    src/grayscale.cu
    src/convolution_naive.cu
    src/convolution_tiled.cu
)
target_link_libraries(image_accel PUBLIC CUDA::cudart)

add_executable(correctness_tests tests/correctness_tests.cpp)
target_link_libraries(correctness_tests PRIVATE image_accel CUDA::cudart)

add_executable(benchmark_runner benchmarks/benchmark.cpp)
target_link_libraries(benchmark_runner PRIVATE image_accel CUDA::cudart)

# Added Streams Benchmark Executable
add_executable(benchmark_streams benchmarks/benchmark_streams.cu)
target_link_libraries(benchmark_streams PRIVATE image_accel CUDA::cudart)
