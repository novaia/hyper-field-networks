#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float data[16];
} mat4;

mat4 get_perspective_matrix(float fov, float near_plane, float far_plane, float aspect_ratio);
mat4 get_x_rotation_matrix(float angle);
mat4 get_y_rotation_matrix(float angle);

#ifdef __cplusplus
} // extern "C"
#endif
