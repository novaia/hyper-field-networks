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

image_t* load_png(const char* file_name);
void save_frame_to_png(const char* filename, unsigned int width, unsigned int height);
void save_depth_to_png(const char* filename, unsigned int width, unsigned int height);

void save_multi_view_transforms_json(
    const float fov_x, const float fov_y,
    const unsigned int num_views, const float* euler_angles,
    const char* file_name
);

typedef struct 
{ 
    unsigned int num_vertices;
    float* vertices; 
    float* normals;
    float* texture_coords;
} obj_t;

obj_t* load_obj(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_normals,
    const unsigned int max_indices
);

#ifdef __cplusplus
} // extern "C"
#endif
