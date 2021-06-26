# Content
- [Content](#content)
- [CUDA-NVIDIA_Learning](#cuda-nvidia_learning)
- [Lab1 - Hello world](#lab1---hello-world)
- [Lab2 - Matrix Multiplication](#lab2---matrix-multiplication)
- [Lab3 - Median Filter](#lab3---median-filter)
# CUDA-NVIDIA_Learning
Programming Massively Parallel Processors with CUDA-NVIDIA
 - Run with command (to compile codes): 
```
nvcc [all of *.cu] -o [name_file_output] -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio
```

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
   - 
# Lab1 - Hello world

# Lab2 - Matrix Multiplication

# Lab3 - Median Filter