# Content
- [Content](#content)
  - [- MyMedianFilter](#--mymedianfilter)
- [CUDA-NVIDIA_Learning (On Jetson Nano)](#cuda-nvidia_learning-on-jetson-nano)
- [Lab1 - Hello world](#lab1---hello-world)
- [Lab2 - Matrix Multiplication](#lab2---matrix-multiplication)
- [Lab3 - Median Filter](#lab3---median-filter)
  - [MyMedianFilter](#mymedianfilter)
-----------------------
# CUDA-NVIDIA_Learning (On Jetson Nano)
Programming Massively Parallel Processors with CUDA-NVIDIA on Jetson Nano Kit
 - Using nvprof tool: 
```
sudo /usr/local/cuda-10.2/bin/nvprof --metrics all ./[name_file_output]
```
 - On target before debugging: 
```
sudo chmod a+rw /dev/nvhost-dbg-gpu
```
  - check its L4T info "JetPack never installs to the Jetson. Only L4T or packages, so there isn’t any JetPack version associated with the Jetson other than possibly the install GUI front end was JetPack of a certain version. On the other hand, so far as I know, there is only one L4T version associated with each JetPack release.":
```
cat /etc/nv_tegra_release
``` 
   - To find a file in a current path:
```
find [path_dictionary_name] -name [file_name]
```
  - To query the properties of the CUDA devices present in the system:
```
cd /usr/local/cuda-10.2/samples/1_Utilities/deviceQuery
./deviceQuery
```
**Note:** 
 1. Dynamically Linked "Shared Object" Libraries(.so) is found in **/usr/lib/aarch64-linux-gnu/** on Jetson Nano. Example this "libcublas.so" library will be linked to compiler with a flag "-lcublas". 
# Lab1 - Hello world

# Lab2 - Matrix Multiplication
  - Run with command (to compile code using cublas library):
  ```
  nvcc matrixMulCUBLAS.cpp -o out -I/usr/local/cuda-10.2/samples/common/inc -I/usr/include/ -lcublas
  ```

# Lab3 - Median Filter
## MyMedianFilter
 - Run with command (to compile codes): 
```
nvcc [all of *.cu or *.cpp] -o [name_file_output] -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -Xptxas="-v" 
```
 - You can customize the parameters which include KERNEL_SIZE, NUM_ELEMENTS (= KERNEL_SIZE ^ 2), TILE_SIZE, ITER_NUM, ... in a _Common.hpp_ header file.
1. Example for running code with NotShareMem_Main.cu: 
``` 
nvcc -gencode arch=compute_53,code=sm_53 MedianFilter.cu NotSharedMem_Main.cu ReadingImage.cu -o out -I/usr/include/opencv4 -lopencv_core  -lopencv_imgcodecs
``` 
2. Example for running code with ShareMem_Main.cu:
```
nvcc -gencode arch=compute_53,code=sm_53 MedianFilter.cu SharedMem_Main.cu ReadingImage.cu -o outSharedMem -I/usr/include/opencv4 -lopencv_core  -lopencv_imgcodecs
```

