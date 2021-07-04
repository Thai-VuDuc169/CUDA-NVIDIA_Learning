#include <stdlib.h>
#include <stdio.h>

__global__ void mallocTest()
{
    size_t size = 123;
    char* ptr = (char*)malloc(size);
    memset(ptr, 0, size);
    ptr[0] = 9;
    printf("Thread %d got pointer: %p: %d\n", threadIdx.x, ptr, ptr[0]);
    free(ptr);
}

// int main()
// {
//     // Set a heap size of 128 megabytes. Note that this must
//     // be done before any kernel is launched.
//     cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
//     mallocTest<<<1, 5>>>();
//     cudaDeviceSynchronize();
//     return 0;
    
// }
#include<iostream>

int main(){
    int r= 2;
    int arr[r];
    arr[0]= 3;
    std::cout<< arr[0];
}