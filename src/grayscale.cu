#include "common.hpp"

__global__ void grayscaleKernel(const unsigned char* __restrict__ rgb, 
                               unsigned char* __restrict__ gray, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char r = rgb[idx * 3 + 0];
        unsigned char g = rgb[idx * 3 + 1];
        unsigned char b = rgb[idx * 3 + 2];
        gray[idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void launchGrayscale(const unsigned char* d_in, unsigned char* d_out, int width, int height, dim3 blockDim) {
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    grayscaleKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
}
