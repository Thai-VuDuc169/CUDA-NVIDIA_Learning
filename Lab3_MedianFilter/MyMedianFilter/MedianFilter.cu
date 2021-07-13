#include "MedianFilter.hpp"

struct Point
{
   int x;
   int y;
   // constructors
   // __device__ Point()
   // {
   //    this->x = 0;
   //    this->y = 0;
   // }

   // __device__ Point(int input_x, int input_y)
   // {
   //    this->x = input_x;
   //    this->y = input_y;
   // }
};

__global__ 
void actMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols)
/*******************\
input: mảng chứa ảnh 1 chiều đã thực hiện padding.
output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
\*******************/
{
   const int row = blockIdx.y * blockDim.y + threadIdx.y;
   const int col = blockIdx.x * blockDim.x + threadIdx.x;
   int pad_num = (int)(KERNEL_SIZE/2);
   const int num_elements = NUM_ELEMENTS;
   int new_cols = old_cols + 2 * pad_num;
   if ((row >= pad_num) && (row <= (pad_num - 1 + old_rows))
         && (col >= pad_num) && (col <= (pad_num - 1 + old_cols)))
   {
      u_char temp_array[num_elements];
      // trích xuất các phần tử trong kernel ra mảng để sắp xếp
      // i -> rows
      for (int i = 0; i < KERNEL_SIZE; i++)
      {
         // j -> cols
         for (int j = 0; j < KERNEL_SIZE; j++)
            {
               temp_array[i * KERNEL_SIZE + j] = input[((row-pad_num) + i) * new_cols + ((col-pad_num) + j)];
            }
      }
      // Ascending the array and replace pixel to the output image
      for(int i = 0; i < num_elements - 1; i++)
      {
         for (int j = i + 1; j < num_elements; j++)
         {
            if (temp_array[i] > temp_array[j])
            {
               u_char swap = temp_array[i];
               temp_array[i] = temp_array[j];
               temp_array[j] = swap;
            }
            if (i > (int)((num_elements)/2))
            {
               output[(row-pad_num) * old_cols + (col-pad_num)] = temp_array[(int)((num_elements)/2)];
               return;
            }
         }
      }
   }
};

__global__
void actMedianFilterUsingSharedMem(u_char *input, u_char *output, int old_rows, int old_cols)
/*******************\
input: mảng chứa ảnh 1 chiều đã thực hiện padding.
output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
\*******************/
{
   const int thx = threadIdx.x;
   const int thy = threadIdx.y;
   const int row = blockIdx.y * blockDim.y + thy;
   const int col = blockIdx.x * blockDim.x + thx;

   const int pad_num = (int)(KERNEL_SIZE/2);
   const int num_elements = NUM_ELEMENTS;
   const int input_cols = old_cols + 2 * pad_num;
   const int input_rows = old_rows + 2 * pad_num;
   const int NUM_THREADS = TILE_SIZE * TILE_SIZE;

   // a share mem in each block
   __shared__ u_char shared_mem [TILE_SIZE + pad_num * 2] [TILE_SIZE + pad_num * 2];
   // Khởi tạo điểm tham chiếu tọa độ tại điểm đầu của mỗi shrared mem block so với tọa độ thực nằm ở input
   __shared__ Point starting_point_ref_shmem2input;
   starting_point_ref_shmem2input.x = blockIdx.x * TILE_SIZE; 
   starting_point_ref_shmem2input.y = blockIdx.y * TILE_SIZE;
   // __syncthreads();

   // tính toán số vòng lặp tối đa của các thread sau (TILE_SIZE*TILE_SIZE) để thực hiện nạp các phần tử từ input vào Shared Mem 
   int iter_of_threads;
   {
      float coefficient = 1.0 + (2.0 * pad_num) / TILE_SIZE;
      float temp_iter = coefficient * coefficient;
      if (temp_iter == float(int(temp_iter)))
         iter_of_threads = int(temp_iter);
      else 
         iter_of_threads = int(temp_iter) + 1;
   }
   Point temp_point_ref2input;
   // load data from input to shared mem
   for (int i = 0; i < iter_of_threads; i++)
   {
      // định vị trí hiện tại của từng vị trí ứng với ID thread (tính bằng "thy * TILE_SIZE + thx") tại mỗi vị trí trí lặp trên Shared Mem
      // vị trí này là tương đối so với mỗi shared mem (lấy điểm đầu tiên của mỗi shared mem làm tham chiếu - "starting_point_ref_shmem2input") 
      int shmem_y = ((thy * TILE_SIZE + thx) + 1 + i * NUM_THREADS) / (TILE_SIZE + 2*pad_num);
      int shmem_x = ((thy * TILE_SIZE + thx) + 1 + i * NUM_THREADS) - shmem_y * (TILE_SIZE + 2*pad_num);
      if (shmem_x == 0)
      {
         shmem_y = shmem_y - 1; shmem_x = TILE_SIZE + 2*pad_num - 1; 
      }
      else
      { 
         shmem_x = shmem_x - 1; 
      }
      // kiểm tra chỉ số shmem_x và shmem_y có vượt ra ngoài shared mem không ? nếu có thì ko thực hiện tiếp
      if (shmem_x > TILE_SIZE + 2*pad_num - 1 || shmem_y > TILE_SIZE + 2*pad_num - 1)
         continue;
         
      // dựa vào starting_point_ref_shmem2input để xác định vị trí hiện tại của điểm trong input
      // Point temp_point_ref2input((starting_point_ref_shmem2input.x + shmem_x), (starting_point_ref_shmem2input.y + shmem_y));
      temp_point_ref2input.x = starting_point_ref_shmem2input.x + shmem_x;
      temp_point_ref2input.y = starting_point_ref_shmem2input.y + shmem_y;

      // kiểm tra xem có phải là nằm ngoài input hay không (nằm ngoài input thì không thực hiện tiếp)
      if (temp_point_ref2input.x > input_cols - 1 || temp_point_ref2input.y > input_rows - 1)
         continue;
      shared_mem[shmem_x][shmem_y] = input[temp_point_ref2input.y * input_cols + temp_point_ref2input.x];
   }
   // đảm bảo tất cả data đều nằm trong shread_mem của mỗi block trước khi tiếp tục
   __syncthreads();
   // kiểm tra các thread ứng với xử lý mỗi điểm ảnh của output có nằm ở ngoài ảnh output không?
   if (row >= old_rows || col >= old_cols)
      return;   

   // trích xuất các phần tử trong mỗi kernel tại mỗi thread ứng với mỗi điểm ảnh của output
   u_char temp_array[num_elements]; // được lưu trong L2 cache của mỗi thread ?
   for(int i = 0; i < KERNEL_SIZE; i++)
   {
      for (int j = 0; j < KERNEL_SIZE; j++)
      {
         temp_array[i * KERNEL_SIZE + j] = shared_mem[thx + j][thy + i];
      }
   }
   // sap xep cac phan tu va thay the vao output
   for(int i = 0; i < num_elements - 1; i++)
   {
      for (int j = i + 1; j < num_elements; j++)
      {
         if ( temp_array[i] > temp_array[j] )
         {
            u_char swap = temp_array[i];
            temp_array[i] = temp_array[j];
            temp_array[j] = swap;
         }
         if (i > (int)((num_elements)/2))
         {
            output[row * old_cols + col] = temp_array[(int)(num_elements/2)];
            return;
         }
      }
   }
};