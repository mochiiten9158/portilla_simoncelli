#define STB_IMAGE_IMPLEMENTATION        // <-- add this
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdint.h>
#include <stdlib.h>

void normalize_and_save_png(const char *filename, float *img, int width, int height) {
    float min_val = img[0], max_val = img[0];
    for (int i = 1; i < width * height; i++) {
        if (img[i] < min_val) min_val = img[i];
        if (img[i] > max_val) max_val = img[i];
    }
    float range = (max_val - min_val);
    if (range < 1e-8f) range = 1.0f;
    uint8_t *img8 = (uint8_t *)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        float val = (img[i] - min_val) / range;
        img8[i] = (uint8_t)(255.0f * val);
    }
    stbi_write_png(filename, width, height, 1, img8, width);
    free(img8);
}