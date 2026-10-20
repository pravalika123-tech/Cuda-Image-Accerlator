#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define TILE_W 16
#define TILE_H 16
#define FILTER_RADIUS 1
#define FILTER_DIM 3

void launchGrayscale(const unsigned char* d_in, unsigned char* d_out, int width, int height, dim3 blockDim);
void launchConvNaive(const unsigned char* d_in, unsigned char* d_out, int width, int height, 
                     const float* d_kernel, int kSize, dim3 blockDim);
void launchConvTiled(const unsigned char* d_in, unsigned char* d_out, int width, int height, 
                     const float* d_kernel, int kSize, dim3 blockDim);
