#include <stdio.h>
#include <iostream>
#include "Median_Filter.h"
#include "Bitmap.h"
#include <ctime>
#include "Common.hpp"
#include "ReadingImage.hpp"

#define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/sp_noise.jpg"
// #define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/lena512.bmp"
// #define KERNEL_SIZE 3 // assert >= 0
#define TILE_SIZE 4

const int window_size = WINDOW_SIZE;
// #define ITERATIONS ( 1 )

int main()
{
	int kernel_size = 3;
   Matrix *input_mat = new Matrix(INPUT_IMAGE_PATH, kernel_size);
   Matrix *output_mat = new Matrix(input_mat->rows, input_mat->cols);

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)input_mat->cols / (float)TILE_SIZE),
		(int)ceil((float)input_mat->rows / (float)TILE_SIZE));
	
	// size_t heapsize = sizeof(u_char) * 9 * (int)((int)ceil((float)input_mat->cols / (float)TILE_SIZE) * TILE_SIZE)
	// 		* (int)((int)ceil((float)input_mat->rows / (float)TILE_SIZE) * TILE_SIZE);
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize);	  
	// medianFilterKernel<<<dimGrid, dimBlock>>>(input_mat->d_elements, output_mat->d_elements, 
	// 													input_mat->cols, input_mat->rows);
	actMedianFilter<<<dimGrid, dimBlock>>>(input_mat->d_elements, output_mat->d_elements, 
											kernel_size, input_mat->rows, input_mat->cols);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	output_mat->copyCudaMemoryD2H();
	// save output image
   output_mat->saveImage("Filted_Image_v1");


   delete input_mat, output_mat;
   return 0;
};