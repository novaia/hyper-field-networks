#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrices.h"

inline double degrees_to_radians(double degrees)
{
    const double pi = 3.14159265358979323846;
    return degrees * (pi / 180.0);
}

inline mat4 mat4mul(mat4 a, mat4 b)
{
    mat4 result = { .data = {0.0f} };
    for(unsigned int col = 0; col < 4; col++)
    {
        const unsigned int col_offset = col * 4;
        for(unsigned int row = 0; row < 4; row++)
        {
            for(unsigned int i = 0; i < 4; i++)
            {
                result.data[col_offset + row] += a.data[i * 4 + row] * b.data[col_offset + i];
            }
        }
    }
    return result;
}

inline void vec3mat4mul(
    mat4 matrix, float in_x, float in_y, float in_z, 
    float* out_x, float* out_y, float* out_z
){
    const float in_w = 1.0f;
    float out_x_temp = 0.0f;
    float out_y_temp = 0.0f;
    float out_z_temp = 0.0f;
    float out_w_temp = 0.0f;
    for(unsigned int row = 0; row < 4; row++)
    {
        const unsigned int row_offset = row * 4;
        const float col_x = matrix.data[row_offset];
        const float col_y = matrix.data[row_offset + 1];
        const float col_z = matrix.data[row_offset + 2];
        const float col_w = matrix.data[row_offset + 3];
        switch(row)
        {
            
            case 0:
                out_x_temp = (in_x * col_x) + (in_y * col_y) + (in_z * col_z) + (in_w * col_w);
                break;
            case 1:
                out_y_temp = (in_x * col_x) + (in_y * col_y) + (in_z * col_z) + (in_w * col_w);
                break;
            case 2:
                out_z_temp = (in_x * col_x) + (in_y * col_y) + (in_z * col_z) + (in_w * col_w);
                break;
            case 3:
                out_w_temp = (in_x * col_x) + (in_y * col_y) + (in_z * col_z) + (in_w * col_w);
                break;
        }
    }
    *out_x = out_x_temp / out_w_temp;
    *out_y = out_y_temp / out_w_temp;
    *out_z = out_z_temp / out_w_temp;
}

inline mat4 get_perspective_matrix(
    float fov, float near_plane, float far_plane, float aspect_ratio
){
    const float tan_half_fov = (float)tan(degrees_to_radians((double)fov) / 2.0);
    mat4 perspective_matrix = { 
        .data = {
            1.0f / (aspect_ratio * tan_half_fov), 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f / tan_half_fov, 0.0f, 0.0f,
            0.0f, 0.0f, -(far_plane + near_plane) / (far_plane - near_plane), -1.0f,
            0.0f, 0.0f, -2.0f * far_plane * near_plane / (far_plane - near_plane), 0.0f
        }
    };
    return perspective_matrix;
}

extern inline mat4 get_orthogonal_matrix(
    float left, float right, float bottom, float top, float near_plane, float far_plane
){
    float width = right - left;
    float height = top - bottom;
    float depth = far_plane - near_plane;
    
    mat4 orthogonal_matrix = {
        .data = {
            2.0f / width, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / height, 0.0f, 0.0f,
            0.0f, 0.0f, -2.0f / depth, 0.0f,
            -(right + left) / width, -(top + bottom) / (height), -(far_plane + near_plane) / depth, 1.0f
        }
    };
    return orthogonal_matrix;
}

inline mat4 get_x_rotation_matrix(float angle)
{
    const float theta = (float)degrees_to_radians((double)angle);
    const float cos_theta = (float)cos(theta);
    const float sin_theta = (float)sin(theta);
    mat4 x_rotation_matrix = {
        .data = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, cos_theta, sin_theta, 0.0f, 
            0.0f, -sin_theta, cos_theta, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    };
    return x_rotation_matrix;
}

inline mat4 get_y_rotation_matrix(float angle)
{
    const float theta = (float)degrees_to_radians((double)angle);
    const float cos_theta = (float)cos(theta);
    const float sin_theta = (float)sin(theta);
    mat4 y_rotation_matrix = {
        .data = {
            cos_theta, 0.0f, -sin_theta, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f, 
            sin_theta, 0.0f, cos_theta, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    };
    return y_rotation_matrix;
}

inline mat4 get_model_matrix(
    float position_x, float position_y, float position_z, 
    float rotation_x, float rotation_y, float rotation_z
){
    const mat4 x_rotation_matrix = get_x_rotation_matrix(rotation_x);
    const mat4 y_rotation_matrix = get_y_rotation_matrix(rotation_y);
    mat4 model_matrix = mat4mul(x_rotation_matrix, y_rotation_matrix);
    model_matrix.data[MAT4_X_TRANSLATION_INDEX] = position_x;
    model_matrix.data[MAT4_Y_TRANSLATION_INDEX] = position_y;
    model_matrix.data[MAT4_Z_TRANSLATION_INDEX] = position_z;
    return model_matrix;
}

inline mat4 get_model_matrix_from_rotation(
    const float rotation_x, const float rotation_y, const float rotation_z, const float zoom
){
    const mat4 x_rotation_matrix = get_x_rotation_matrix(rotation_x);
    const mat4 y_rotation_matrix = get_y_rotation_matrix(rotation_y);
    mat4 model_matrix = mat4mul(x_rotation_matrix, y_rotation_matrix);
    float translation_x = 0.0f, translation_y = 0.0f, translation_z = 0.0f;
    vec3mat4mul(model_matrix, 0.0f, 0.0f, zoom, &translation_x, &translation_y, &translation_z);
    model_matrix.data[MAT4_X_TRANSLATION_INDEX] = translation_x;
    model_matrix.data[MAT4_Y_TRANSLATION_INDEX] = translation_y;
    model_matrix.data[MAT4_Z_TRANSLATION_INDEX] = translation_z;
    return model_matrix;
}

inline mat4 get_lookat_matrix_from_rotation(
    float rotation_x, float rotation_y, float rotation_z, float zoom
){
    const float theta_x = (float)degrees_to_radians((double)rotation_x);
    const float theta_y = (float)degrees_to_radians((double)rotation_y);
    const float cos_x = cosf(theta_x);
    const float sin_x = sinf(theta_x);
    const float cos_y = cosf(theta_y);
    const float sin_y = sinf(theta_y);
    
    mat4 matrix = {
        .data = {
            cos_y, sin_x * sin_y, -cos_x * sin_y, 0.0f,
            0.0f, cos_x, sin_x, 0.0f,
            sin_y, -sin_x * cos_y, cos_x * cos_y, 0.0f,
            0.0f, 0.0f, -zoom, 1.0f
        }
    };
    return matrix;
}
