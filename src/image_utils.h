#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

// stb_image reader - declaration only (implementation is in image_utils.cpp)
#include "external/stb_image.h"

void normalize_and_save_png(const char *filename, float *img, int width, int height);

#endif