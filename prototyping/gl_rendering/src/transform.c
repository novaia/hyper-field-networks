#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transform.h"

static inline double degrees_to_radians(double degrees)
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
    model_matrix[MAT4_X_TRANSLATION_INDEX] = position_x;
    model_matrix[MAT4_Y_TRANSLATION_INDEX] = position_y;
    model_matrix[MAT4_Z_TRANSLATION_INDEX] = position_z;
    return model_matrix;
}
