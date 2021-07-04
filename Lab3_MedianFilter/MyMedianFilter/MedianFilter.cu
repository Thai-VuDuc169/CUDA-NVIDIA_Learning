#include "MedianFilter.hpp"


__global__ 
void actPaddingMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols)
/*******************\
input: mảng chứa ảnh 1 chiều đã thực hiện padding.
output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
\*******************/
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
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

// __global__
// void __global__ void actMedianFilter(u_char *input, u_char *output, int old_rows, int old_cols)
// /*******************\
// input: mảng chứa ảnh 1 chiều không có padding.
// output: mảng chứa ảnh 1 chiều sau khi thực hiện median filter.
// rows, cols: kích thước chiều cao, rộng của ảnh đầu ra.
// \*******************/
// {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    int pad_num = (int)(KERNEL_SIZE/2);


// };

// __global__
// void actMedianFilterSharedKernel(u_char *input, u_char *output, int old_rows, int old_cols)
// {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    const int pad_num = (int)(KERNEL_SIZE/2);
//    __shared__ u_char shemem[(TILE_SIZE + pad_num*2)][(TILE_SIZE + pad_num*2)];
   
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
                                                                    smem[tx_l+1][ty_l+1] =                           Input_Image[ty_g*Image_Width + tx_g];      // --- center
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