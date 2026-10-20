#include "common.hpp"

#define SMEM_W (TILE_W + 2 * FILTER_RADIUS)
#define SMEM_H (TILE_H + 2 * FILTER_RADIUS)

__global__ void convTiledKernel(const unsigned char* __restrict__ in, 
                                unsigned char* __restrict__ out, 
                                int width, int height, 
                                const float* __restrict__ kernel, int kSize) {
    __shared__ unsigned char s_data[SMEM_H][SMEM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outX = blockIdx.x * blockDim.x + tx;
    int outY = blockIdx.y * blockDim.y + ty;

    int smemX = tx + FILTER_RADIUS;
    int smemY = ty + FILTER_RADIUS;

    int clampedX = min(max(outX, 0), width - 1);
    int clampedY = min(max(outY, 0), height - 1);
    s_data[smemY][smemX] = in[clampedY * width + clampedX];

    if (ty < FILTER_RADIUS) {
        int topY = min(max(outY - FILTER_RADIUS, 0), height - 1);
        s_data[ty][smemX] = in[topY * width + clampedX];

        int botY = min(max(outY + blockDim.y, 0), height - 1);
        s_data[smemY + blockDim.y][smemX] = in[botY * width + clampedX];
    }

    if (tx < FILTER_RADIUS) {
        int leftX = min(max(outX - FILTER_RADIUS, 0), width - 1);
        s_data[smemY][tx] = in[clampedY * width + leftX];

        int rightX = min(max(outX + blockDim.x, 0), width - 1);
        s_data[smemY][smemX + blockDim.x] = in[clampedY * width + rightX];
    }

    if (tx < FILTER_RADIUS && ty < FILTER_RADIUS) {
        int cX = min(max(outX - FILTER_RADIUS, 0), width - 1);
        int cY = min(max(outY - FILTER_RADIUS, 0), height - 1);
        s_data[ty][tx] = in[cY * width + cX];

        cX = min(max(outX + blockDim.x, 0), width - 1);
        cY = min(max(outY - FILTER_RADIUS, 0), height - 1);
        s_data[ty][smemX + blockDim.x] = in[cY * width + cX];

        cX = min(max(outX - FILTER_RADIUS, 0), width - 1);
        cY = min(max(outY + blockDim.y, 0), height - 1);
        s_data[smemY + blockDim.y][tx] = in[cY * width + cX];

        cX = min(max(outX + blockDim.x, 0), width - 1);
        cY = min(max(outY + blockDim.y, 0), height - 1);
        s_data[smemY + blockDim.y][smemX + blockDim.x] = in[cY * width + cX];
    }

    __syncthreads();

    if (outX < width && outY < height) {
        float sum = 0.0f;
        for (int kr = -FILTER_RADIUS; kr <= FILTER_RADIUS; ++kr) {
            for (int kc = -FILTER_RADIUS; kc <= FILTER_RADIUS; ++kc) {
                sum += kernel[(kr + FILTER_RADIUS) * kSize + (kc + FILTER_RADIUS)] 
                       * s_data[smemY + kr][smemX + kc];
            }
        }
        out[outY * width + outX] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
    }
}

void launchConvTiled(const unsigned char* d_in, unsigned char* d_out, int width, int height, 
                     const float* d_kernel, int kSize, dim3 blockDim) {
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    convTiledKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, d_kernel, kSize);
    CUDA_CHECK(cudaGetLastError());
}
