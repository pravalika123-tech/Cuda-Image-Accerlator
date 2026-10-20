#include "common.hpp"
#include "cpu_reference.hpp"
#include <cassert>
#include <random>

void runCorrectnessTest() {
    int w = 512, h = 512;
    std::vector<unsigned char> h_rgb(w * h * 3);
    std::vector<unsigned char> h_gray_cpu(w * h);
    std::vector<unsigned char> h_gray_gpu(w * h);
    
    std::mt19937 rng(42);
    for (auto& val : h_rgb) val = rng() % 256;

    cpuGrayscale(h_rgb.data(), h_gray_cpu.data(), w, h);

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, w * h * 3));
    CUDA_CHECK(cudaMalloc(&d_out, w * h));
    CUDA_CHECK(cudaMemcpy(d_in, h_rgb.data(), w * h * 3, cudaMemcpyHostToDevice));

    launchGrayscale(d_in, d_out, w, h, dim3(16, 16));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gray_gpu.data(), d_out, w * h, cudaMemcpyDeviceToHost));

    int max_diff = 0;
    for (int i = 0; i < w * h; ++i) {
        max_diff = std::max(max_diff, std::abs(h_gray_cpu[i] - h_gray_gpu[i]));
    }
    std::cout << "[Test: Grayscale] Max difference: " << max_diff;
    assert(max_diff <= 1 && "Grayscale output exceeds numerical tolerance!");
    std::cout << " -> PASSED\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    runCorrectnessTest();
    return 0;
}
