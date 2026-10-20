#include "common.hpp"
#include "cpu_reference.hpp"
#include <iomanip>

void runBenchmark() {
    const std::vector<int> sizes = {512, 1024, 2048, 4096};
    const float kernel3x3[9] = {
        1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16
    };

    std::cout << std::left 
              << std::setw(8)  << "Size"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "HtoD (ms)"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "Tiled (ms)"
              << std::setw(12) << "DtoH (ms)"
              << std::setw(14) << "Speedup(K)"
              << std::setw(14) << "Speedup(E2E)" << "\n";
    std::cout << std::string(98, '-') << "\n";

    for (int N : sizes) {
        size_t bytes = N * N * sizeof(unsigned char);
        std::vector<unsigned char> h_in(N * N, 128);
        std::vector<unsigned char> h_out_cpu(N * N);
        std::vector<unsigned char> h_out_gpu(N * N);

        auto t0 = std::chrono::high_resolution_clock::now();
        cpuConvolution(h_in.data(), h_out_cpu.data(), N, N, kernel3x3, 3);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        unsigned char *d_in, *d_out;
        float *d_k;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMalloc(&d_k, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_k, kernel3x3, 9 * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t startH2D, stopH2D, startK, stopK, startTiled, stopTiled, startD2H, stopD2H;
        CUDA_CHECK(cudaEventCreate(&startH2D));
        CUDA_CHECK(cudaEventCreate(&stopH2D));
        CUDA_CHECK(cudaEventCreate(&startK));
        CUDA_CHECK(cudaEventCreate(&stopK));
        CUDA_CHECK(cudaEventCreate(&startTiled));
        CUDA_CHECK(cudaEventCreate(&stopTiled));
        CUDA_CHECK(cudaEventCreate(&startD2H));
        CUDA_CHECK(cudaEventCreate(&stopD2H));

        launchConvTiled(d_in, d_out, N, N, d_k, 3, dim3(TILE_W, TILE_H));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(startH2D));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stopH2D));

        CUDA_CHECK(cudaEventRecord(startK));
        launchConvNaive(d_in, d_out, N, N, d_k, 3, dim3(TILE_W, TILE_H));
        CUDA_CHECK(cudaEventRecord(stopK));

        CUDA_CHECK(cudaEventRecord(startTiled));
        launchConvTiled(d_in, d_out, N, N, d_k, 3, dim3(TILE_W, TILE_H));
        CUDA_CHECK(cudaEventRecord(stopTiled));

        CUDA_CHECK(cudaEventRecord(startD2H));
        CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stopD2H));
        CUDA_CHECK(cudaDeviceSynchronize());

        float h2d_ms, naive_ms, tiled_ms, d2h_ms;
        cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D);
        cudaEventElapsedTime(&naive_ms, startK, stopK);
        cudaEventElapsedTime(&tiled_ms, startTiled, stopTiled);
        cudaEventElapsedTime(&d2h_ms, startD2H, stopD2H);

        float e2e_ms = h2d_ms + tiled_ms + d2h_ms;
        float speedup_kernel = cpu_ms / tiled_ms;
        float speedup_e2e = cpu_ms / e2e_ms;

        std::cout << std::left 
                  << std::setw(8)  << N
                  << std::setw(12) << cpu_ms
                  << std::setw(12) << h2d_ms
                  << std::setw(14) << naive_ms
                  << std::setw(14) << tiled_ms
                  << std::setw(12) << d2h_ms
                  << std::setw(14) << speedup_kernel
                  << std::setw(14) << speedup_e2e << "\n";

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_k));
    }
}

int main() {
    runBenchmark();
    return 0;
}
