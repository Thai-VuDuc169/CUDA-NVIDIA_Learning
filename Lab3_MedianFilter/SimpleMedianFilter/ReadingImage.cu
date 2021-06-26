#include "ReadingImage.hpp"

Matrix::Matrix()
{
   this->rows = 0;
   this->cols = 0;
   this->d_elements = NULL;
   this->h_elements = NULL;
};

Matrix::Matrix(const int &temp_row, const int &temp_col)
{
   this->rows = temp_row;
   this->cols = temp_col;
   size_t mat_size = this->rows * this->cols * sizeof(u_char);
   this->h_elements = (u_char*)malloc(mat_size);
   assert(this->h_elements != NULL);
   gpuErrchk( cudaMalloc(&this->d_elements, mat_size) );
};

Matrix::Matrix(const cv::Mat &mat)
{
   this->rows = mat.rows;
   this->cols = mat.cols;
   size_t mat_size = this->rows * this->cols * sizeof(u_char);
   this->h_elements = (u_char*)malloc(mat_size);
   assert(this->h_elements != NULL);
   for (int i = 0; i < this->rows * this->cols; i++)
      this->h_elements[i] = mat.data[i];
   gpuErrchk(cudaMalloc(&this->d_elements, mat_size));
   gpuErrchk(cudaMemcpy(this->d_elements, this->h_elements, mat_size, cudaMemcpyHostToDevice));
};

// hàm tạo khởi tạo ma trận sau khi đã padding
Matrix::Matrix(const std::string &path_name, const int &kernel_size)
{
   assert(kernel_size % 2 != 0 && kernel_size >= 0); // kernel_size phải là số lẻ dương
   cv::Mat mat = cv::imread(path_name, cv::IMREAD_GRAYSCALE);
   this->rows = mat.rows;
   this->cols = mat.cols;
   int pad_num = (int)(kernel_size/2);
   int new_rows = this->rows + pad_num * 2;
   int new_cols = this->cols + pad_num * 2;
   size_t mat_size = new_rows * new_cols * sizeof(u_char);
   this->h_elements = (u_char*)malloc(mat_size);
   assert(this->h_elements != NULL);
   for (int i = 0; i < new_rows; i++)
   {
      for (int j = 0; j < new_cols; j++)
      {
         if ((i < pad_num) || (i > (pad_num - 1 + this->rows)) 
               || (j < pad_num) || (j > (pad_num - 1 + this->cols)))
         {
            this->h_elements[i * new_cols + j] = 0;
         }
         else
         {
            this->h_elements[i * new_cols + j] = mat.data[(i - pad_num) * this->cols + (j - pad_num)];
         }
      }
   }
   gpuErrchk(cudaMalloc(&this->d_elements, mat_size));
   gpuErrchk(cudaMemcpy(this->d_elements, this->h_elements, mat_size, cudaMemcpyHostToDevice));
};

Matrix::~Matrix()
{
   gpuErrchk( cudaFree(this->d_elements) );
   // delete[] this->h_elements; 
   // Aborted (core dumped): giải phóng con trỏ nhiều lần dẫn đến lỗi này, khi một đối tượng giải phóng, nó sẽ tự động giải phóng toàn bộ con trỏ (chứa data) trong nó
};

u_char Matrix::getMatElement(const int &temp_row, const int &temp_col)
{
   assert(this->h_elements != NULL);
   return this->h_elements[temp_row * this->cols + temp_col];
};

void Matrix::setMatElement(const int &temp_row, const int &temp_col, const int &val)
{
   assert(this->h_elements != NULL);
   this->h_elements[temp_row * this->cols + temp_col] = val;
};

std::ostream& operator<< (std::ostream &out, Matrix mat)
{
   assert(mat.h_elements != NULL);
   out<< "\n===========INFORMATION OF THIS MATRIX==========="
      << "\n|The number of rows: " << mat.rows
      << "\n|The number of cols: " << mat.cols
      << "\n================================================";
   return out;
};

void Matrix::saveImage(const std::string &str)
{
   cv::Mat temp_mat(this->rows, this->cols, CV_8UC1, this->h_elements);
   bool result = false;
   try
   {
      result = imwrite(str + std::string(".jpg"), temp_mat);
   }
   catch (const cv::Exception& ex)
   {
      std::cerr << "Exception converting image to JPG format: " << ex.what() << std::endl;
   }
   if (result)
      std::cout << "Saved JPG file." << std::endl;
   else
      std::cerr << "ERROR: Can't save JPG file." << std::endl;
};

void Matrix::copyCudaMemoryD2H()
{
   size_t mat_size = this->rows * this->cols * sizeof(u_char);
   gpuErrchk( cudaMemcpy(this->h_elements, this->d_elements, mat_size, cudaMemcpyDeviceToHost) );
};