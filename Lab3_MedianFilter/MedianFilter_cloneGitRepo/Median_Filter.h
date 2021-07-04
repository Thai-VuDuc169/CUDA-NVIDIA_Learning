#pragma once

#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H
#include "Bitmap.h"
#include <cuda_runtime.h>
#ifndef WINDOW_SIZE
#define WINDOW_SIZE (3)
#endif

void MedianFilterCPU(Bitmap* image, Bitmap* outputImage);
bool MedianFilterGPU(Bitmap* image, Bitmap* outputImage, bool sharedMemoryUse);
__global__ void medianFilterKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight);
__global__ void medianFilterSharedKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight);
__global__ void actMedianFilter(u_char *input, u_char *output, int kernel_size, int old_rows, int old_cols);

#endif