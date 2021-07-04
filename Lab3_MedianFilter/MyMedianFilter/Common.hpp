#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include<cstdio> 
#include<cuda_runtime.h>

#define KERNEL_SIZE 7 // assert >= 0
#define NUM_ELEMENTS 49
#define TILE_SIZE 16
#define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/sp_noise.jpg"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort= true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s (line: %d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif
