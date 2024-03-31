#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct
{
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    float* pixels;
} image_t;

typedef struct
{
    image_t* texture;
} material_t;

typedef struct 
{ 
    uint32_t num_vertices;
    float* vertices; 
    float* normals;
    float* texture_coords;
    material_t* material;
} mesh_t;

#ifdef __cplusplus
} // extern "C"
#endif
