#include <stdint.h>

#pragma once
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

typedef struct
{
    image_t* texture;
} material_t;

typedef struct 
{ 
    unsigned int num_vertices;
    float* vertices; 
    float* normals;
    float* texture_coords;
} obj_t;

#ifdef __cplusplus
} // extern "C"
#endif
