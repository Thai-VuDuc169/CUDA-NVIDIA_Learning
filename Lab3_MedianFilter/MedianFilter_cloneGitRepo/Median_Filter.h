#pragma once

#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H
#include "Bitmap.h"
#ifndef WINDOW_SIZE
#define WINDOW_SIZE (3)
#endif

void MedianFilterCPU(Bitmap* image, Bitmap* outputImage);
bool MedianFilterGPU(Bitmap* image, Bitmap* outputImage, bool sharedMemoryUse);
#endif