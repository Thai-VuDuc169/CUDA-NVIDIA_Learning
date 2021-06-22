/*******************\
The Task List:
1. 

Author: Vu Duc Thai
\*******************/
#include"ReadingImage.hpp"
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
// #include<opencv2/highgui.hpp>
#include<iostream>
#include<typeinfo>

#define INPUT_IMAGE_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/test1.jpg"
#define KERNEL_SIZE 5

// using namespace cv;

// __global__ void actMedianFilter(u_char *intput, u_char *output, 
                           // int kernel_size, int rows, int cols)
/*******************\
input: mảng chứa ảnh 1 chiều đã thực hiện padding.
output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
kernel_size: kích thước bộ lọc (this is a odd num).
rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
\*******************/
// {

// };


int main()
{
   Matrix *input_mat = new Matrix(INPUT_IMAGE_PATH, KERNEL_SIZE);
   Matrix *output_mat = new Matrix(input_mat->rows, input_mat->cols);

   std::cout << "=============test1======" << std::endl;
   std::cout << "The input matrix infor: " << *input_mat << std::endl;
   std::cout << "The output matrix infor: " << *output_mat << std::endl;
   // std::cout << "sdaf: " << (int)input_mat->h_elements[100] << std::endl;
   // std::cout << "the test element value: " << typeid(input_mat->h_elements).name() << std::endl; // kieemr tra xem input_mat->h_elements có bằng null ko 
   // std::cout << "=============test2======" << std::endl;


   // Set our CTA and Grid dimensions
   // int threads = 32;
   // int blocks1 = (N + threads - 1) / threads; // định nghĩa theo số chiều của ảnh sau khi padding
   // int blocks1 = (N + threads - 1) / threads;
   // // Setup our kernel launch parameters
   // dim3 NUM_THREADS (threads, threads); 
   // dim3 NUM_BLOCKS (blocks, blocks);
   // // Launch our kernel
   // <<<NUM_BLOCKS, NUM_THREADS>>>(mat1, mat2, result_mat, N);
   // cudaDeviceSynchronize();

   delete input_mat, output_mat;
   // delete a;
   return 0;
}


