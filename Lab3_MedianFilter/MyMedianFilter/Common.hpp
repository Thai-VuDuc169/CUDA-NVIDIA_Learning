#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cstdio> 
#include <cuda_runtime.h>
#include <dirent.h>
#include <vector>
#include <string>

// ==================================== customable ==================================== 
#define KERNEL_SIZE 11 // assert >= 0      
#define NUM_ELEMENTS 121       
#define TILE_SIZE 32
#define ITER_NUM 50
#define INPUT_FOLDER_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/LabImageTest/Noised/"        
#define OUTPUT_FOLDER_PATH "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/LabImageTest/Filtered_SharedMem/"
#define FILE_NAME "/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/LabImageTest/Filtered_SharedMem/benchmark_log_JetsonNano_SharedMem_MedianFilter.txt"
// ==================================== customable ====================================

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort= true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s (file: %s, line: %d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) 
{
   DIR *p_dir = opendir(p_dir_name);
   if (p_dir == nullptr) 
   {
      return -1;
   }
   struct dirent* p_file = nullptr;
   while ((p_file = readdir(p_dir)) != nullptr) 
   {
      if (strcmp(p_file->d_name, ".") != 0 &&
         strcmp(p_file->d_name, "..") != 0) 
      {
         std::string cur_file_name(p_file->d_name);
         file_names.push_back(cur_file_name);
      }
   }
   closedir(p_dir);
   return 0;
};

#endif
