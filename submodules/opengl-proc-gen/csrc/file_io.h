#pragma once

#include <n3dc/n3dc_obj.h>
#include "vector_matrix_math.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    unsigned int width;
    unsigned int height;
    unsigned int stride;
    float* pixels;
} image_t;

image_t* load_png(const char* file_name);
void save_frame_to_png(const char* filename, unsigned int width, unsigned int height);
void save_depth_to_png(const char* filename, unsigned int width, unsigned int height);

void save_multi_view_transforms_json(
    const float fov_x, const float fov_y,
    const unsigned int num_views, const mat4* transform_matrices,
    const char* file_name, const int with_depth
);

#ifdef __cplusplus
} // extern "C"
#endif
