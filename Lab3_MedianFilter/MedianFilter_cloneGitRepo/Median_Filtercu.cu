#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "Median_Filter.h"
#include <time.h>
#include <cuda_runtime.h>
#define TILE_SIZE 4

__global__ void medianFilterKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char filterVector[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	// size_t size_mat = 9 * sizeof(u_char);
	// u_char *filterVector = (u_char*) malloc(9 * sizeof(u_char)) ;
	// cudaMalloc((void**)&filterVector, 9 * sizeof(u_char));
	if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
		outputImageKernel[row * imageWidth + col] = 0;
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++) {
				filterVector[x * WINDOW_SIZE + y] = inputImageKernel[(row + x - 1) * imageWidth + (col + y - 1)];

			}
		}
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImageKernel[row * imageWidth + col] = filterVector[4];
		// cudaFree (filterVector);
	}
	// free(filterVector);
}

__global__ void medianFilterSharedKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned char sharedmem[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
	bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE - 1);
	bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE - 1);

	if (is_x_left)
		sharedmem[threadIdx.x][threadIdx.y + 1] = 0;
	else if (is_x_right)
		sharedmem[threadIdx.x + 2][threadIdx.y + 1] = 0;
	if (is_y_top) {
		sharedmem[threadIdx.x + 1][threadIdx.y] = 0;
		if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = 0;
		else if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y] = 0;
	}
	else if (is_y_bottom) {
		sharedmem[threadIdx.x + 1][threadIdx.y + 2] = 0;
		if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y + 2] = 0;
		else if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y + 2] = 0;
	}

	//Setup pixel value
	sharedmem[threadIdx.x + 1][threadIdx.y + 1] = inputImageKernel[row * imageWidth + col];
	if (is_x_left && (col > 0))
		sharedmem[threadIdx.x][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col - 1)];
	else if (is_x_right && (col < imageWidth - 1))
		sharedmem[threadIdx.x + 2][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col + 1)];
	if (is_y_top && (row > 0)) {
		sharedmem[threadIdx.x + 1][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + col];
		if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col - 1)];
		else if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col + 1)];
	}
	else if (is_y_bottom && (row < imageHeight - 1)) {
		sharedmem[threadIdx.x + 1][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + col];
		if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col + 1)];
		else if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col - 1)];
	}

	__syncthreads();

	unsigned char filterVector[9] = { sharedmem[threadIdx.x][threadIdx.y], sharedmem[threadIdx.x + 1][threadIdx.y], sharedmem[threadIdx.x + 2][threadIdx.y],
				   sharedmem[threadIdx.x][threadIdx.y + 1], sharedmem[threadIdx.x + 1][threadIdx.y + 1], sharedmem[threadIdx.x + 2][threadIdx.y + 1],
				   sharedmem[threadIdx.x][threadIdx.y + 2], sharedmem[threadIdx.x + 1][threadIdx.y + 2], sharedmem[threadIdx.x + 2][threadIdx.y + 2] };
	{
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImageKernel[row * imageWidth + col] = filterVector[4];
	}
}

bool MedianFilterGPU(Bitmap* image, Bitmap* outputImage, bool sharedMemoryUse) {
	//Cuda error and image values.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaError_t status;
	int width = image->Width();
	int height = image->Height();

	int size = width * height * sizeof(char);
	//initialize images.
	unsigned char* deviceinputimage;
	cudaMalloc((void**)&deviceinputimage, size);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) <<
			std::endl;
		return false;
	}
	cudaMemcpy(deviceinputimage, image->image, size, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed for cudaMemcpy cudaMemcpyHostToDevice: " << cudaGetErrorString(status) <<
			std::endl;
		cudaFree(deviceinputimage);
		return false;
	}
	unsigned char* deviceOutputImage;
	cudaMalloc((void**)&deviceOutputImage, size);
	//take block and grids.
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)image->Width() / (float)TILE_SIZE),
		(int)ceil((float)image->Height() / (float)TILE_SIZE));

	//Check for shared memories and call the kernel
	if (!sharedMemoryUse)
		medianFilterKernel << <dimGrid, dimBlock >> > (deviceinputimage, deviceOutputImage, width, height);
	else
		medianFilterSharedKernel << <dimGrid, dimBlock >> > (deviceinputimage, deviceOutputImage, width, height);



	// save output image to host.
	cudaMemcpy(outputImage->image, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	status = cudaGetLastError();



	if (status != cudaSuccess) {
		std::cout << "Kernel failed for cudaMemcpy cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) <<
			std::endl;
		cudaFree(deviceinputimage);
		cudaFree(deviceOutputImage);
		return false;
	}
	//Free the memory
	cudaFree(deviceinputimage);
	cudaFree(deviceOutputImage);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	printf("time %f\n", time);
	return true;
}

#define NUM_ELEMENTS 9
__global__ 
void actMedianFilter(u_char *input, u_char *output, int kernel_size, int old_rows, int old_cols)
/*******************\
input: mảng chứa ảnh 1 chiều đã thực hiện padding.
output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
kernel_size: kích thước bộ lọc (this is a odd num).
rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
\*******************/
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int pad_num = (int)(kernel_size/2);
   const int num_elements = NUM_ELEMENTS;
   int new_cols = old_cols + 2 * pad_num;
   if ((row >= pad_num) && (row <= (pad_num - 1 + old_rows))
         && (col >= pad_num) && (col <= (pad_num - 1 + old_cols)))
   {
		u_char temp_array[num_elements];
		// u_char *temp_array = (u_char*)malloc(num_elements * sizeof(u_char));
      // memset(temp_array, 0 , num_elements * sizeof(u_char));
      // u_char *temp_array = new u_char[num_elements]; 
      // trích xuất các phần tử trong kernel ra mảng để sắp xếp
      // i -> rows
      for (int i = 0; i < kernel_size; i++)
      {
         // j -> cols
         for (int j = 0; j < kernel_size; j++)
            {
               temp_array[i * kernel_size + j] = input[((row-pad_num) + i) * new_cols + ((col-pad_num) + j)];
            }
      }
      // Ascending the array
      for(int i = 0; i < num_elements - 1; i++)
      {
         for (int j = i + 1; j < num_elements; j++)
         {
            if (temp_array[i] > temp_array[j])
            {
               u_char swap = temp_array[i];
               temp_array[i] = temp_array[j];
               temp_array[j] = swap;
            }
         }
      }
      // replace pixel to the ouput image
      output[(row-pad_num) * old_cols + (col-pad_num)] = temp_array[(int)((num_elements)/2)];
      // deallocate the memory
      // free(temp_array);
      // delete[] temp_array;
   }
};
