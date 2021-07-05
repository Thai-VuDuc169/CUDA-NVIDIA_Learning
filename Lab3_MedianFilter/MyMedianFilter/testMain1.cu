/*******************\
The Task List:
1. 
Author: Vu Duc Thai
\*******************/
#include "ReadingImage.hpp"
#include "MedianFilter.hpp"
#include <cstdio>
// #include 

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>



int main()
{
   Matrix *input_mat = new Matrix("/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/sp_noise.jpg", KERNEL_SIZE);
   // std::cout << *input_mat << std::endl;
   Matrix *output_mat = new Matrix(input_mat->rows, input_mat->cols);
   // std::cout << *output_mat << std::endl;
   //the number of elements for padding matrix
   int new_rows = input_mat->rows + (int)(KERNEL_SIZE/2) * 2;
   int new_cols = input_mat->cols + (int)(KERNEL_SIZE/2) * 2;
   // int true_size = input_mat->rows * input_mat->cols;
   // Set our CTA and Grid dimensions
   dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)new_cols / (float)TILE_SIZE),
		(int)ceil((float)new_rows / (float)TILE_SIZE));
   actPaddingMedianFilter <<<dimGrid, dimBlock>>> (input_mat->d_elements, output_mat->d_elements, input_mat->rows, input_mat->cols);

   // size_t shmem_size = (16+2) * (16+2) * blocks * blocks * sizeof(u_char); 
   // cudaFuncSetCacheConfig(Optimized_Kernel_Function_shared, cudaFuncCachePreferShared);
   // Optimized_Kernel_Function_shared <<<NUM_BLOCKS, NUM_THREADS, shmem_size>>> (input_mat->d_elements, output_mat->d_elements, 
   //                                                 input_mat->rows, input_mat->cols);

   // medianFilterSharedKernel <<<NUM_BLOCKS, NUM_THREADS>>> (input_mat->d_elements, output_mat->d_elements, 
   //                                                 input_mat->rows, input_mat->cols);
   gpuErrchk(cudaPeekAtLastError());
   gpuErrchk(cudaDeviceSynchronize());
   // copy data về host memory
   output_mat->copyCudaMemoryD2H();
   // save output image
   output_mat->saveImage("Filted_Image_v3");

   delete input_mat, output_mat;
   return 0;
}
