#ifndef _MEDIAN_FILTER_HPP_
#define _MEDIAN_FILTER_HPP_
#include<cuda_runtime.h>
#include "Common.hpp"

struct Point;
__global__  void actMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols); // input có padding
__global__ void actMedianFilterUsingSharedMem(u_char *input, u_char *output, int old_rows, int old_cols); // tối ưu sử dụng shared mem

#endif