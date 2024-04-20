#pragma once

#define MAT4_THIRD_COLUMN_START (4 * 2)
#define MAT4_X_TRANSLATION_INDEX 12
#define MAT4_Y_TRANSLATION_INDEX 13
#define MAT4_Z_TRANSLATION_INDEX 14

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float data[16];
} mat4;

extern inline double degrees_to_radians(double degrees);
extern inline mat4 mat4mul(mat4 a, mat4 b);
extern inline mat4 get_perspective_matrix(
    float fov, float near_plane, float far_plane, float aspect_ratio
);
extern inline mat4 get_orthogonal_matrix(
    float left, float right, float bottom, float top, float near_plane, float far_plane
);
extern inline mat4 get_x_rotation_matrix(float angle);
extern inline mat4 get_y_rotation_matrix(float angle);
extern inline mat4 get_model_matrix(
    float position_x, float position_y, float position_z, 
    float rotation_x, float rotation_y, float rotation_z
);
extern inline mat4 get_model_matrix_from_rotation(
    const float rotation_x, const float rotation_y, const float rotation_z, const float zoom
);
extern inline mat4 get_lookat_matrix_from_rotation(
    float rotation_x, float rotation_y, float rotation_z, float zoom
);

#ifdef __cplusplus
} // extern "C"
#endif
