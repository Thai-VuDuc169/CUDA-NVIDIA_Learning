/*******************\
The Task List:
1. Calculating matrix multiplication in CUDA from scratch
2. Mesure execution time & Bandwidth (Run on CPUs and GPUs)
3. Write back to *.txt format files (kernel_matrix_size, execution_time, device_name)
4. Plot Benchmarks (using python to plot) 
Author: Vu Duc Thai
\*******************/
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include <helper_functions.h>
#include <helper_cuda.h>

using std::cout;
using std::endl;
using std::chrono::steady_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

__global__ void multiMatries(int *mat1, int *mat2, int *result_mat, int N) // không thể sử dụng "const" or "&", -> error "segmatation fault (core dumped)"
{
   // Calculate the global row and column for each thread
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   // Boundary check for our matrix
   int temp = 0;
   for (int i = 0 ; i < N; i++)
   {
      temp += mat1[row*N + i] * mat2[i*N + col];
   }
   // Write back the result
   result_mat[row*N + col] = temp;
}

// Initialize a square matrix with random numbers between 1-> 100
void initMatrix(int *mat, const int &N)
{
   for (int i = 0; i < N * N; i++)
   {
      mat[i] = rand() % 100 + 1;
   }
}

// verify a result from mat1 * mat2 with a ref_mat 
void verifyMatries(int *mat1, int *mat2, int *ref_mat, const int &N)
{
   int temp;
   // i -> rows
   for (int i = 0; i < N; i++)
   {
      // j -> columns
      for (int j = 0; j < N; j++)
      {
         temp = 0;
         for (int k = 0; k < N; k++)
         {
            temp += mat1[i*N + k] * mat2[k*N + j];
         }
         assert(temp == ref_mat[i*N + j]);
      }
   }
}
int main()
{
   // Set our square matrix dimension (2^10 x 2^10 default)
   int N = 1 << 12; 
   size_t bytes = N * N * sizeof(int);
   // Allocate memory for our matrices
   int *mat1, *mat2, *result_mat;
   // Allocate memory for our matries
   cudaMallocManaged(&mat1, bytes);
   cudaMallocManaged(&mat2, bytes);
   cudaMallocManaged(&result_mat, bytes);
   // Initialize our matrices
   initMatrix(mat1, N);
   initMatrix(mat2, N);
   // Set our CTA and Grid dimensions
   int threads = 32;
   int blocks = (N + threads - 1) / threads;
   // Setup our kernel launch parameters
   dim3 NUM_THREADS (threads, threads); 
   dim3 NUM_BLOCKS (blocks, blocks);

   // measure time using GPU timer
   cudaEvent_t start, stop;
   // Allocate CUDA events that we'll use for timing
   checkCudaErrors(cudaEventCreate(&start));
   checkCudaErrors(cudaEventCreate(&stop));

   auto start_CPU = steady_clock::now();
   cudaEventRecord(start, NULL);
   // Launch our kernel
   multiMatries<<<NUM_BLOCKS, NUM_THREADS>>>(mat1, mat2, result_mat, N);
   cudaDeviceSynchronize();
   auto end_CPU = steady_clock::now();
   
   cudaEventRecord(stop, NULL);
   float msecTotal = 0.0f;
   cudaEventElapsedTime(&msecTotal, start, stop);

   cout << "Execution time when running on GPU: "
         << duration_cast <duration<double>>(end_CPU - start_CPU).count()
         << " seconds.\n\tGPU timer: " << msecTotal << endl;
   



   start_CPU = steady_clock::now();
   // verify the result of matrix multiplication
   // verifyMatries(mat1, mat2, result_mat, N);
   end_CPU = steady_clock::now();
   cout << "Execution time when running on CPU: "
         << duration_cast <duration<double>>(end_CPU - start_CPU).count()
         << " seconds." << endl;

   cout << "Successful" << endl;
   // Free unified memory 
   cudaFree(mat1);
   cudaFree(mat2);
   cudaFree(result_mat);
   return 0;
}