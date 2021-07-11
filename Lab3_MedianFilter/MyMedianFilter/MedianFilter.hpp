#ifndef _MEDIAN_FILTER_HPP_
#define _MEDIAN_FILTER_HPP_
#include<cuda_runtime.h>
#include "Common.hpp"

struct Point;
__global__  void actPaddingMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols); // input có padding
// __global__ void actMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols); // input không có padding
__global__ void actMedianFilterSharedKernel(u_char *input, u_char *output, int old_rows, int old_cols); // tối ưu sử dụng shared mem

__global__ void medianFilterSharedKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight);
__global__ void Optimized_Kernel_Function_shared(u_char *Input_Image, u_char *Output_Image, int Image_Width, int Image_Height);


#endif