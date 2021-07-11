#include "MedianFilter.hpp"

struct Point
{
   int x;
   int y;
};

__global__ 
void actPaddingMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols)
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
void actMedianFilterSharedKernel(u_char *input, u_char *output, int old_rows, int old_cols)
{
   const int thx = threadIdx.x;
   const int thy = threadIdx.y;
   const int row = blockIdx.y * blockDim.y + thy;
   const int col = blockIdx.x * blockDim.x + thx;
   const int pad_num = (int)(KERNEL_SIZE/2);

   const int num_elements = NUM_ELEMENTS;
   const int input_cols = old_cols + 2 * pad_num;
   const int input_rows = old_rows + 2 * pad_num;
   // printf("\ninput_cols = %d, input_rows = %d", input_cols, input_rows);

   const int NUM_THREADS = TILE_SIZE * TILE_SIZE;

   // share mem in each block
   __shared__ u_char shared_mem [TILE_SIZE + pad_num * 2] [TILE_SIZE + pad_num * 2];
   __shared__ Point starting_point_ref_shared2input;
   starting_point_ref_shared2input.x = blockIdx.x * TILE_SIZE; 
   starting_point_ref_shared2input.y = blockIdx.y * TILE_SIZE;
   __syncthreads();
   // printf("\n===starting_point_ref_shared2input.x = %d, starting_point_ref_shared2input.y = %d", 
   //          starting_point_ref_shared2input.x, starting_point_ref_shared2input.y);

   // tinh toan so vong lap
   // int iter_of_threads = (int)ceil((1+ 2*pad_num/TILE_SIZE) * (1+ 2*pad_num/TILE_SIZE));
   int iter_of_threads = 2;
   // load data from input to shared mem
   for (int i = 0; i < iter_of_threads; i++)
   {
      // dinh vi tri toa do trong shared mem
      int y = ((thy * TILE_SIZE + thx) + 1 + i * NUM_THREADS) / (TILE_SIZE + 2*pad_num);
      int x = ((thy * TILE_SIZE + thx) + 1 + i * NUM_THREADS) - y * (TILE_SIZE + 2*pad_num);
      if (x == 0)
         { y = y - 1; x = TILE_SIZE + 2*pad_num - 1; }
      else if (y == 0)
         { x = x - 1; }
      else
         { x = x - 1; }

      // if (x < 0 || y < 0)
      // {
      //    printf("\nerror -1: in shared mem");
      //    printf( "\nx= %.3d, y= %.3d", x, y);
      //    continue;
      // }
      if (x > TILE_SIZE + 2*pad_num - 1 || y > TILE_SIZE + 2*pad_num - 1)
      {
         // printf("\nerror -1: out of boundary of shared mem");
         continue;
      }
      // printf( "\n===========\nx= %.3d, y= %.3d", x, y);
      // dua vao starting_point_ref_shared2input de xac dinh vi tri trong anh input
      int input_y = starting_point_ref_shared2input.y + y;
      int input_x = starting_point_ref_shared2input.x + x;
      // kiem tra xem co phai la bien hay khong (nam ngoai input thi gan bang 0, hoac continue cho nhanh)
      if (input_x > input_cols - 1 || input_y > input_rows - 1)
         continue;

      // if (input_x < 0 || input_y < 0)
      // {
      //    printf("\nerror -1: in global of input");
         // printf( "\ninput_x= %.3d, input_y= %.3d", input_x, input_y);
      // }   
      shared_mem[x][y] = input[input_y * input_cols + input_x];
      // printf ("\n=========\ninput_x= %.3d, input_y= %.3d \nshared_mem[y][x]= %d", 
      //          input_x, input_y, shared_mem[y][x]);
   }
   __syncthreads();
   // kiem tra output o ngoai anh cua output ko?
   if (row > input_rows || col > input_cols)
      return;   

   // trich xuat cac phan tu trong moi kernel
   u_char temp_array[num_elements];
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
            // printf("\nrow= %d, col= %d, array[middle] = %d", row, col, temp_array[(int)(num_elements/2)]); 
            return;
         }
      }
   }
};
// {
//    const int row = blockIdx.y * blockDim.y + threadIdx.y;
//    const int col = blockIdx.x * blockDim.x + threadIdx.x;
//    const int pad_num = (int)(KERNEL_SIZE/2);
//    const int num_elements = NUM_ELEMENTS;
//    const int new_cols = old_cols + 2 * pad_num;
//    __shared__ u_char shared_mem[TILE_SIZE] [TILE_SIZE] [NUM_ELEMENTS];
//    if ((row >= pad_num) && (row <= (pad_num - 1 + old_rows))
//          && (col >= pad_num) && (col <= (pad_num - 1 + old_cols)))
//    {
//       // load each element in a kernel to share mem
//       // i -> rows
//       for (int i = 0; i < KERNEL_SIZE; i++)
//       {
//          // j -> cols
//          for (int j = 0; j < KERNEL_SIZE; j++)
//          {
//             shared_mem[threadIdx.x][threadIdx.y][i * KERNEL_SIZE + j] 
//                   = input[(row - pad_num + i) * new_cols + (col - pad_num + j)];
//          }
//       }
//       // Ascending the array and replace pixel to the output image
//       for(int i = 0; i < num_elements - 1; i++)
//       {
//          for (int j = i + 1; j < num_elements; j++)
//          {
//             if ( shared_mem[threadIdx.x][threadIdx.y][i] > shared_mem[threadIdx.x][threadIdx.y][j] )
//             {
//                u_char swap = shared_mem[threadIdx.x][threadIdx.y][i];
//                shared_mem[threadIdx.x][threadIdx.y][i] = shared_mem[threadIdx.x][threadIdx.y][j];
//                shared_mem[threadIdx.x][threadIdx.y][j] = swap;
//             }
//             if (i > (int)((num_elements)/2))
//             {
//                output[row * old_cols + col] 
//                      = shared_mem[threadIdx.x][threadIdx.y][(int)(num_elements/2)];
//                return;
//             }
//          }
//       }
//    }
// };


__global__ 
void medianFilterSharedKernel(unsigned char* inputImageKernel, unsigned char* outputImageKernel, int imageWidth, int imageHeight) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned char sharedmem[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
	bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE - 1);
	bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE - 1);

	if (is_x_left)
		sharedmem[threadIdx.x][threadIdx.y + 1] = 0;
	else if (is_x_right)
		sharedmem[threadIdx.x + 2][threadIdx.y + 1] = 0;
	if (is_y_top) {
		sharedmem[threadIdx.x + 1][threadIdx.y] = 0;
		if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = 0;
		else if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y] = 0;
	}
	else if (is_y_bottom) {
		sharedmem[threadIdx.x + 1][threadIdx.y + 2] = 0;
		if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y + 2] = 0;
		else if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y + 2] = 0;
	}

	//Setup pixel value
	sharedmem[threadIdx.x + 1][threadIdx.y + 1] = inputImageKernel[row * imageWidth + col];
	if (is_x_left && (col > 0))
		sharedmem[threadIdx.x][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col - 1)];
	else if (is_x_right && (col < imageWidth - 1))
		sharedmem[threadIdx.x + 2][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col + 1)];
	if (is_y_top && (row > 0)) {
		sharedmem[threadIdx.x + 1][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + col];
		if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col - 1)];
		else if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col + 1)];
	}
	else if (is_y_bottom && (row < imageHeight - 1)) {
		sharedmem[threadIdx.x + 1][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + col];
		if (is_x_right)
			sharedmem[threadIdx.x + 2][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col + 1)];
		else if (is_x_left)
			sharedmem[threadIdx.x][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col - 1)];
	}

	__syncthreads();

	unsigned char filterVector[9] = { sharedmem[threadIdx.x][threadIdx.y], sharedmem[threadIdx.x + 1][threadIdx.y], sharedmem[threadIdx.x + 2][threadIdx.y],
				   sharedmem[threadIdx.x][threadIdx.y + 1], sharedmem[threadIdx.x + 1][threadIdx.y + 1], sharedmem[threadIdx.x + 2][threadIdx.y + 1],
				   sharedmem[threadIdx.x][threadIdx.y + 2], sharedmem[threadIdx.x + 1][threadIdx.y + 2], sharedmem[threadIdx.x + 2][threadIdx.y + 2] };
	{
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImageKernel[row * imageWidth + col] = filterVector[4];
	}
};

#define BLOCK_HEIGHT 16
#define BLOCK_WIDTH 16

__global__ 
void Optimized_Kernel_Function_shared(u_char *Input_Image, u_char *Output_Image, int Image_Width, int Image_Height)
{
    const int tx_l = threadIdx.x;                           // --- Local thread x index
    const int ty_l = threadIdx.y;                           // --- Local thread y index

    const int tx_g = blockIdx.x * blockDim.x + tx_l;        // --- Global thread x index
    const int ty_g = blockIdx.y * blockDim.y + ty_l;        // --- Global thread y index

    __shared__ u_char smem[BLOCK_WIDTH+2][BLOCK_HEIGHT+2];

    // --- Fill the shared memory border with zeros
    if (tx_l == 0)                      smem[tx_l]  [ty_l+1]    = 0;    // --- left border
    else if (tx_l == BLOCK_WIDTH-1)     smem[tx_l+2][ty_l+1]    = 0;    // --- right border
    if (ty_l == 0) {                    smem[tx_l+1][ty_l]      = 0;    // --- upper border
        if (tx_l == 0)                  smem[tx_l]  [ty_l]      = 0;    // --- top-left corner
        else if (tx_l == BLOCK_WIDTH-1) smem[tx_l+2][ty_l]      = 0;    // --- top-right corner
        }   else if (ty_l == BLOCK_HEIGHT-1) {smem[tx_l+1][ty_l+2]  = 0;    // --- bottom border
        if (tx_l == 0)                  smem[tx_l]  [ty_l+2]    = 0;    // --- bottom-left corder
        else if (tx_l == BLOCK_WIDTH-1) smem[tx_l+2][ty_l+2]    = 0;    // --- bottom-right corner
    }

    // --- Fill shared memory
                                                                    smem[tx_l+1][ty_l+1] =    Input_Image[ty_g*Image_Width + tx_g];      // --- center
    if ((tx_l == 0)&&((tx_g > 0)))                                      smem[tx_l]  [ty_l+1] = Input_Image[ty_g*Image_Width + tx_g-1];      // --- left border
    else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))         smem[tx_l+2][ty_l+1] = Input_Image[ty_g*Image_Width + tx_g+1];      // --- right border
    if ((ty_l == 0)&&(ty_g > 0)) {                                      smem[tx_l+1][ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g];    // --- upper border
            if ((tx_l == 0)&&((tx_g > 0)))                                  smem[tx_l]  [ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g-1];  // --- top-left corner
            else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))     smem[tx_l+2][ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g+1];  // --- top-right corner
         } else if ((ty_l == BLOCK_HEIGHT-1)&&(ty_g < Image_Height - 1)) {  smem[tx_l+1][ty_l+2] = Input_Image[(ty_g+1)*Image_Width + tx_g];    // --- bottom border
         if ((tx_l == 0)&&((tx_g > 0)))                                 smem[tx_l]  [ty_l+2] = Input_Image[(ty_g-1)*Image_Width + tx_g-1];  // --- bottom-left corder
        else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))     smem[tx_l+2][ty_l+2] = Input_Image[(ty_g+1)*Image_Width + tx_g+1];  // --- bottom-right corner
    }
    __syncthreads();

    // --- Pull the 3x3 window in a local array
    u_char v[9] = { smem[tx_l][ty_l],   smem[tx_l+1][ty_l],     smem[tx_l+2][ty_l],
                            smem[tx_l][ty_l+1], smem[tx_l+1][ty_l+1],   smem[tx_l+2][ty_l+1],
                            smem[tx_l][ty_l+2], smem[tx_l+1][ty_l+2],   smem[tx_l+2][ty_l+2] };    

    // --- Bubble-sort
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (v[i] > v[j]) { // swap?
                u_char tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
         }
    }

    // --- Pick the middle one
    Output_Image[ty_g*Image_Width + tx_g] = v[4];
}