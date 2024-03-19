#pragma once

typedef struct
{
    float data[16];
} mat4;

mat4 get_perspective_matrix(float fov, float near_plane, float far_plane, float aspect_ratio);
