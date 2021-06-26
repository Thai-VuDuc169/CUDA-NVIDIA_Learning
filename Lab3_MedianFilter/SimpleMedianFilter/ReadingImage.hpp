#ifndef _READING_IMAGE_H_
#define _READING_IMAGE_H_
// #include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include"Common.hpp"
#include<string>
#include<cassert>
#include<iostream>

// using namesapce cv;
using namespace std;

class Matrix
{
   public:
      int cols, rows;
      u_char* d_elements;
      u_char* h_elements;
   public:
      // Constructor
      Matrix();
      Matrix(const int &temp_row, const int &temp_col);
      Matrix(const cv::Mat &mat);
      Matrix(const string &path_name, const int &kernel_size); // Ma trận khởi tạo có sẵn padding ứng với kích thước của kerenl_size
      // Destructor
      ~Matrix();
   public:
      // Methods extract element
      u_char getMatElement(const int &temp_row, const int &temp_col);
      void setMatElement(const int &temp_row, const int &temp_col, const int &val);
      // Methods come with openCV to process image
      // void loadImage();
      void saveImage(const std::string &str);
      // Methods interact with CUDA memory
      void copyCudaMemoryD2H();
   public:
      friend ostream& operator<< (ostream &out, Matrix mat);
};

#endif
