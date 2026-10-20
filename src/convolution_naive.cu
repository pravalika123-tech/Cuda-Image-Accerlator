#include "common.hpp"

__global__ void convNaiveKernel(const unsigned char* __restrict__ in, 
                                unsigned char* __restrict__ out, 
                                int width, int height, 
                                const float* __restrict__ kernel, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int rad = kSize / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int kr = -rad; kr <= rad; ++kr) {
            for (int kc = -rad; kc <= rad; ++kc) {
                int r = min(max(y + kr, 0), height - 1);
                int c = min(max(x + kc, 0), width - 1);
                sum += kernel[(kr + rad) * kSize + (kc + rad)] * in[r * width + c];
            }
        }
        out[y * width + x] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
    }
}

void launchConvNaive(const unsigned char* d_in, unsigned char* d_out, int width, int height, 
                     const float* d_kernel, int kSize, dim3 blockDim) {
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    convNaiveKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, d_kernel, kSize);
    CUDA_CHECK(cudaGetLastError());
}
