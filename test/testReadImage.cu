/*******************\
The Task List:
1. 

Author: Vu Duc Thai
\*******************/
// #include"ReadingImage.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include<typeinfo>

using namespace cv;

int main()
{
   std::string PATH("/home/thaivu/Projects/CUDA-NVIDIA_Learning/Photo/test1.jpg");
   // std::cout << "hello world" << std::endl;
   // std::cout << "PATH: " << PATH << std::endl;
   Mat img = imread(PATH, IMREAD_GRAYSCALE);
   // u_char *h_mat;
   // size_t mat_size = img.rows * img.cols * sizeof(u_char);
   std::cout << img.rows << " " << img.cols << std::endl;
   // h_mat = (u_char*)malloc(mat_size);
   // this->elements = new u_char[this->rows * this->cols];
   // for(int i = 0; i < img.rows ; i++)
   // {
   //    for(int j = 0; j < img.cols; j++)
   //    {
   //       h_mat[i * img.cols + j] = img.at<u_char>(i, j);
   //    }
   // }
   std::cout << std::hex << (int)(img.data[0 * img.cols + 70]) << std::endl;
   // std::cout << img.at<u_char>(0, 70) << std::endl;
   std::cout << typeid((u_char)(img.data[0 * img.cols + 70])).name() << std::endl;
   // std::cout << typeid(img.at<u_char>(0, 70)).name() << std::endl;
   std::cout << "END" << std::endl;
   // free(h_mat);
   // cudaFree(h_mat);
   return 0;
}


