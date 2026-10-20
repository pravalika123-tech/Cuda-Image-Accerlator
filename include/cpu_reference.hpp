#pragma once
void cpuGrayscale(const unsigned char* rgb, unsigned char* gray, int width, int height);
void cpuConvolution(const unsigned char* in, unsigned char* out, int width, int height, 
                    const float* kernel, int kSize);
void cpuSobel(const unsigned char* in, unsigned char* out, int width, int height);
