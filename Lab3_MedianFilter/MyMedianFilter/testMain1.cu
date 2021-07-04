/*******************\
The Task List:
1. 

Author: Vu Duc Thai
\*******************/
#include"ReadingImage.hpp"
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>

#define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/sp_noise.jpg"
// #define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/test1.jpg"
// #define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/lena512.bmp"
#define KERNEL_SIZE 3 // assert >= 0
#define NUM_ELEMENTS 9
// using namespace cv;

__global__ void actMedianFilter(u_char *input, u_char *output, 
                           int kernel_size, int old_rows, int old_cols)
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
   const int num_elements =  NUM_ELEMENTS;
   int new_cols = old_cols + 2 * pad_num;
   if ((row >= pad_num) && (row <= (pad_num - 1 + old_rows))
               && (col >= pad_num) && (col <= (pad_num - 1 + old_cols)))
   {
      u_char temp_array[num_elements];
      // u_char *temp_array = (u_char*)malloc(num_elements * sizeof(u_char));
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

      // Ascending the array and replace pixel to the output image
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
            if (i > (int)((num_elements)/2))
            {
               output[(row-pad_num) * old_cols + (col-pad_num)] = temp_array[(int)((num_elements)/2)];
               // free (temp_array);
               return;
            }

         }
      }
      // replace pixel to the ouput image
      // output[(row-pad_num) * old_cols + (col-pad_num)] = temp_array[(int)((num_elements)/2)];
      // deallocate the memory
      // free(temp_array);
      // delete[] temp_array;
   }
};


int main()
{
   Matrix *input_mat = new Matrix(INPUT_IMAGE_PATH, KERNEL_SIZE);
   Matrix *output_mat = new Matrix(input_mat->rows, input_mat->cols);

   //the number of elements for padding matrix
   int new_rows = input_mat->rows + (int)(KERNEL_SIZE/2) * 2;
   int new_cols = input_mat->cols + (int)(KERNEL_SIZE/2) * 2;
   int true_size = new_rows * new_cols;
   // Set our CTA and Grid dimensions
   int threads = 32;
   int blocks = (true_size + threads - 1) / threads; // định nghĩa theo số chiều của ảnh sau khi padding
   // Setup our kernel launch parameters
   dim3 NUM_THREADS (threads, threads); 
   dim3 NUM_BLOCKS (blocks, blocks);
   // Launch our kernel
   actMedianFilter<<<NUM_BLOCKS, NUM_THREADS>>>(input_mat->d_elements, output_mat->d_elements, 
                                                KERNEL_SIZE, input_mat->rows, input_mat->cols);
   gpuErrchk(cudaDeviceSynchronize());

      
   // copy data back to host memory
   output_mat->copyCudaMemoryD2H();
   // save output image
   output_mat->saveImage("Filted_Image_v1");

   delete input_mat, output_mat;
   return 0;
}


