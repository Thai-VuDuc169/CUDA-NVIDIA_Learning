/*******************\
The Task List:
1. 

Author: Vu Duc Thai
\*******************/
#include "ReadingImage.hpp"
#include "MedianFilter.hpp"
#include <cstdio>
#include <fstream>  


// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #include <device_launch_parameters.h>

int main()
{
   std::ofstream file_out;
   file_out.open(FILE_NAME, std::ios_base::out | std::ios_base::app );
   file_out << "file_name, rows, cols, kernel_size, bandwidth(GB/s), time_GPU(msec)\n" ;

   std::string input_folder_path(INPUT_FOLDER_PATH);
   std::string output_folder_path(OUTPUT_FOLDER_PATH);
   std::vector<std::string> files;
   read_files_in_dir(input_folder_path.c_str(), files);
   std::string temp_file_path;
   cudaEvent_t start, stop;
   // Allocate CUDA events that we'll use for timing
   gpuErrchk(cudaEventCreate(&start));
   gpuErrchk(cudaEventCreate(&stop));
   for(std::string file : files)
   {
      temp_file_path = input_folder_path + file;
      std::cout << temp_file_path << std::endl;
      std::cout << file << std::endl;
      Matrix *input_mat = new Matrix(temp_file_path, KERNEL_SIZE);
      // std::cout << *input_mat << std::endl; // GPUassert: unspecified launch failure testMain1.cu (line: gpuErrchk(cudaDeviceSynchronize()) )  ????
      Matrix *output_mat = new Matrix(input_mat->rows, input_mat->cols);
      // std::cout << *output_mat << std::endl; // GPUassert: unspecified launch failure testMain1.cu (line: gpuErrchk(cudaDeviceSynchronize()) ) ????
      //the number of elements for padding matrix
      int new_rows = input_mat->rows + (int)(KERNEL_SIZE/2) * 2;
      int new_cols = input_mat->cols + (int)(KERNEL_SIZE/2) * 2;
      // Set our CTA and Grid dimensions
      dim3 dimBlock(TILE_SIZE, TILE_SIZE);
      dim3 dimGrid((int)ceil((float)new_cols / (float)TILE_SIZE),
                     (int)ceil((float)new_rows / (float)TILE_SIZE));
      // size_t shared_mem_size = sizeof(u_char) * TILE_SIZE * TILE_SIZE * NUM_ELEMENTS 
      //    * (int)ceil((float)new_cols / (float)TILE_SIZE) * (int)ceil((float)new_rows / (float)TILE_SIZE);
                     
      // Record the start event
      gpuErrchk(cudaEventRecord(start, NULL));
      
      for(int j = 0; j < ITER_NUM; j++)
      {
         // Launch our kernel
         // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration
         actMedianFilterSharedKernel <<<dimGrid, dimBlock>>> (input_mat->d_elements, output_mat->d_elements, input_mat->rows, input_mat->cols);
         gpuErrchk(cudaPeekAtLastError());
         gpuErrchk(cudaDeviceSynchronize());
      }

      // Record the stop event
      gpuErrchk(cudaEventRecord(stop, NULL));
      // Wait for the stop event to complete
      gpuErrchk(cudaEventSynchronize(stop));
      float msecTotal = 0.0f;
      gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));

      // Compute and print the performance
      float msecPerFilter = msecTotal / ITER_NUM;
      double gigaBytePerFilter = ((double)(input_mat->rows * input_mat->cols)  
                  + (double)(new_rows + new_cols)) * 1 * 1.0e-9f;
      double bandWidth = gigaBytePerFilter / (msecPerFilter / 1000.0f);
      printf( "BandWidth= %.3f, Time= %.3f msec\n", bandWidth, msecPerFilter);
      file_out << std::to_string(KERNEL_SIZE) + "_" + file << ", " << input_mat->rows << ", " << input_mat->cols << ", " << KERNEL_SIZE
                << ", " << bandWidth << ", " << msecPerFilter << "\n";  

      // copy data back to host memory
      output_mat->copyCudaMemoryD2H();
      // save the output image
      output_mat->saveImage(output_folder_path + std::to_string(KERNEL_SIZE) + "_" + file);

      delete input_mat, output_mat;
   }   

   file_out.close();
   std::cout << "===============DONE!================" << std::endl;
   return 0;
}


