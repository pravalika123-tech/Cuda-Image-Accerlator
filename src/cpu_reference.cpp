#include "cpu_reference.hpp"
#include <cmath>
#include <algorithm>

void cpuGrayscale(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        float r = rgb[i * 3 + 0];
        float g = rgb[i * 3 + 1];
        float b = rgb[i * 3 + 2];
        gray[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void cpuConvolution(const unsigned char* in, unsigned char* out, int width, int height, 
                    const float* kernel, int kSize) {
    int rad = kSize / 2;
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int kr = -rad; kr <= rad; ++kr) {
                for (int kc = -rad; kc <= rad; ++kc) {
                    int inR = std::min(std::max(r + kr, 0), height - 1);
                    int inC = std::min(std::max(c + kc, 0), width - 1);
                    float weight = kernel[(kr + rad) * kSize + (kc + rad)];
                    sum += weight * in[inR * width + inC];
                }
            }
            out[r * width + c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
        }
    }
}

void cpuSobel(const unsigned char* in, unsigned char* out, int width, int height) {
    const float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sumX = 0.0f;
            float sumY = 0.0f;
            for (int kr = -1; kr <= 1; ++kr) {
                for (int kc = -1; kc <= 1; ++kc) {
                    int inR = std::min(std::max(r + kr, 0), height - 1);
                    int inC = std::min(std::max(c + kc, 0), width - 1);
                    unsigned char px = in[inR * width + inC];
                    sumX += Gx[(kr + 1) * 3 + (kc + 1)] * px;
                    sumY += Gy[(kr + 1) * 3 + (kc + 1)] * px;
                }
            }
            float mag = std::sqrt(sumX * sumX + sumY * sumY);
            out[r * width + c] = static_cast<unsigned char>(std::min(mag, 255.0f));
        }
    }
}
