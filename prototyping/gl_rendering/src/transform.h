#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define MAT4_X_TRANSLATION_INDEX 12
#define MAT4_Y_TRANSLATION_INDEX 13
#define MAT4_Z_TRANSLATION_INDEX 14

typedef struct
{
    float data[16];
} mat4;

inline mat4 mat4mul(mat4 a, mat4 b);
inline mat4 get_perspective_matrix(
    float fov, float near_plane, float far_plane, float aspect_ratio
);
inline mat4 get_x_rotation_matrix(float angle);
inline mat4 get_y_rotation_matrix(float angle);
inline mat4 get_model_matrix(
    float position_x, float position_y, float position_z, 
    float rotation_x, float rotation_y, float rotation_z
);

#ifdef __cplusplus
} // extern "C"
#endif
