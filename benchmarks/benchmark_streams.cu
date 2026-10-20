#include "common.hpp"
#include <iostream>
#include <vector>
#include <cstring>

// Forward declaration of naive kernel
__global__ void convNaiveKernel(const unsigned char* __restrict__ in, 
                                unsigned char* __restrict__ out, 
                                int width, int height, 
                                const float* __restrict__ kernel, int kSize);

void testStreams(int N = 4096, int numStreams = 4) {
    size_t bytes = N * N * sizeof(unsigned char);
    size_t chunkBytes = bytes / numStreams;
    int chunkHeight = N / numStreams;

    // 1. Allocate page-locked (pinned) memory for true async DMA transfers
    unsigned char *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    std::memset(h_in, 128, bytes);

    unsigned char *d_in, *d_out;
    float *d_k;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, 9 * sizeof(float)));

    const float kernel3x3[9] = {
        1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16
    };
    CUDA_CHECK(cudaMemcpy(d_k, kernel3x3, 9 * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Create streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    convNaiveKernel<<<grid, block>>>(d_in, d_out, N, N, d_k, 3);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Timed Asynchronous Pipeline
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < numStreams; ++i) {
        size_t offset = i * chunkBytes;
        // Asynchronous H2D transfer on stream i
        CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, chunkBytes, 
                                   cudaMemcpyHostToDevice, streams[i]));

        dim3 chunkGrid((N + block.x - 1) / block.x, (chunkHeight + block.y - 1) / block.y);
        
        // Kernel execution on stream i
        convNaiveKernel<<<chunkGrid, block, 0, streams[i]>>>(
            d_in + offset, d_out + offset, N, chunkHeight, d_k, 3
        );

        // Asynchronous D2H transfer on stream i
        CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, chunkBytes, 
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "[Streams Benchmark] 4096 x 4096 (" << numStreams << " streams) End-to-End time: " 
              << elapsed_ms << " ms\n";

    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_k));
}

int main() {
    testStreams(4096, 4);
    return 0;
}
