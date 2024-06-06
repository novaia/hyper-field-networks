#include <math.h>
#include <string.h>
#include "vector_matrix_math.h"

float degrees_to_radians(float degrees)
{
    const float pi = 3.141592653f;
    return degrees * (pi / 180.0f);
}

/* Start of vec3 functions. */
void vec3_copy(const vec3 source, vec3 result)
{
    memcpy(result, source, sizeof(vec3));
}

void vec3_scale(const vec3 v, const float s, vec3 result)
{
    result[0] = v[0] * s;
    result[1] = v[1] * s;
    result[2] = v[2] * s;
}

void vec3_add(const vec3 a, const vec3 b, vec3 result)
{
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
    result[2] = a[2] + b[2];
}

void vec3_mat4_mul(const vec3 v, const mat4 m, vec3 result)
{
    vec3 result_buffer = VEC3_ZERO_INIT;
    for(unsigned int i = 0; i < 3; ++i)
    {
        for(unsigned int k = 0; k < 3; ++k)
        {
            result_buffer[i] += m[k][i] * v[k];
        }
    }
    memcpy(result, result_buffer, sizeof(vec3));
}
/* End of vec3 functions. */

/* Start of vec4 functions. */
float vec4_norm(const vec4 v)
{
    const float x = v[0];
    const float y = v[1];
    const float z = v[2];
    const float w = v[3];
    return sqrtf(x*x + y*y + z*z + w*w);
}
/* End of vec4 functions. */

/* Start of mat4 functions. */
void mat4_copy(const mat4 source, mat4 result)
{
    memcpy(result, source, sizeof(mat4));
}

void mat4_translate(const mat4 m, const vec3 t, mat4 result)
{
    result[3][0] = m[3][0] + t[0];
    result[3][1] = m[3][1] + t[1];
    result[3][2] = m[3][2] + t[2];
}

void mat4_set_translation(mat4 m, const vec3 t)
{
    m[3][0] = t[0];
    m[3][1] = t[1];
    m[3][2] = t[2];
}

void mat4_mul(const mat4 a, const mat4 b, mat4 result)
{
    mat4 result_buffer = MAT4_ZERO_INIT;
    for(unsigned int col = 0; col < 4; ++col)
    {
        for(unsigned int row = 0; row < 4; ++row)
        {
            for(unsigned int i = 0; i < 4; ++i)
            {
                result_buffer[col][row] += a[i][row] * b[col][i];
            }
        }
    }
    memcpy(result, result_buffer, sizeof(mat4));
}

// Scales the first 3 columns of an affine transformation matrix.
void mat4_scale_affine(const mat4 m, const vec3 s, mat4 result)
{
    for(unsigned int col = 0; col < 3; ++col)
    {
        for(unsigned int row = 0; row < 4; ++row)
        {
            result[col][row] = m[col][row] * s[col];
        }
    }
}

void mat4_make_identity(mat4 result)
{
    // Column 0.
    result[0][0] = 1.0f;
    result[0][1] = 0.0f;
    result[0][2] = 0.0f;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = 0.0f;
    result[1][1] = 1.0f;
    result[1][2] = 0.0f;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = 0.0f;
    result[2][1] = 0.0f;
    result[2][2] = 1.0f;
    result[2][3] = 0.0f;
    // Column 3.
    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = 0.0f;
    result[3][3] = 1.0f;
}

void mat4_make_x_rotation(const float degrees, mat4 result)
{
    const float theta = degrees_to_radians(degrees);
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    // Column 0.
    result[0][0] = 1.0f;
    result[0][1] = 0.0f;
    result[0][2] = 0.0f;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = 0.0f;
    result[1][1] = cos_theta;
    result[1][2] = sin_theta;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = 0.0f;
    result[2][1] = -sin_theta;
    result[2][2] = cos_theta;
    result[2][3] = 0.0f;
    // Column 3.
    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = 0.0f;
    result[3][3] = 1.0f;
}

void mat4_make_y_rotation(const float degrees, mat4 result)
{
    const float theta = degrees_to_radians(degrees);
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    // Column 0.
    result[0][0] = cos_theta;
    result[0][1] = 0.0f;
    result[0][2] = -sin_theta;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = 0.0f;
    result[1][1] = 1.0f;
    result[1][2] = 0.0f;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = sin_theta;
    result[2][1] = 0.0f;
    result[2][2] = cos_theta;
    result[2][3] = 0.0f;
    // Column 3.
    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = 0.0f;
    result[3][3] = 1.0f;
}

void mat4_make_z_rotation(const float degrees, mat4 result)
{
    const float theta = degrees_to_radians(degrees);
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    // Column 0.
    result[0][0] = cos_theta;
    result[0][1] = sin_theta;
    result[0][2] = 0.0f;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = -sin_theta;
    result[1][1] = cos_theta;
    result[1][2] = 0.0f;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = 0.0f;
    result[2][1] = 0.0f;
    result[2][2] = 1.0f;
    result[2][3] = 0.0f;
    // Column 3.
    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = 0.0f;
    result[3][3] = 1.0f;
}

void mat4_make_perspective_projection(
    const float fov, const float near_plane, const float far_plane, 
    const float aspect_ratio, mat4 result
){
    float tan_half_fov = tanf(degrees_to_radians(fov) / 2.0f);
    // Column 0.
    result[0][0] = 1.0f / (aspect_ratio * tan_half_fov);
    result[0][1] = 0.0f;
    result[0][2] = 0.0f;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = 0.0f;
    result[1][1] = 1.0f / tan_half_fov;
    result[1][2] = 0.0f;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = 0.0f;
    result[2][1] = 0.0f;
    result[2][2] = -(far_plane + near_plane) / (far_plane - near_plane);
    result[2][3] = -1.0f;
    // Column 3.
    result[3][0] = 0.0f;
    result[3][1] = 0.0f;
    result[3][2] = -2.0f * far_plane * near_plane / (far_plane - near_plane);
    result[3][3] = 0.0f;
}

void mat4_make_orthogonal_projection(
    float left, float right, float bottom, float top, 
    float near_plane, float far_plane, mat4 result
){
    const float width = right - left;
    const float height = top - bottom;
    const float depth = far_plane - near_plane;
    // Column 0.
    result[0][0] = 2.0f / width;
    result[0][1] = 0.0f;
    result[0][2] = 0.0f;
    result[0][3] = 0.0f;
    // Column 1.
    result[1][0] = 0.0f;
    result[1][1] = 2.0f / height;
    result[1][2] = 0.0f;
    result[1][3] = 0.0f;
    // Column 2.
    result[2][0] = 0.0f;
    result[2][1] = 0.0f;
    result[2][2] = -2.0f / depth; 
    result[2][3] = -1.0f;
    // Column 3.
    result[3][0] = -(right + left) / width;
    result[3][1] = -(top + bottom) / height;
    result[3][2] = -(far_plane + near_plane) / depth;
    result[3][3] = 1.0f;
}

void mat4_make_ordinary_model_matrix(const vec3 position, const vec3 rotation, mat4 model_matrix)
{
    mat4 x_rotation_matrix;
    mat4 y_rotation_matrix;
    mat4 z_rotation_matrix;
    mat4_make_x_rotation(rotation[0], x_rotation_matrix);
    mat4_make_y_rotation(rotation[1], y_rotation_matrix);
    mat4_make_z_rotation(rotation[2], z_rotation_matrix);
    // R = R_z * R_y * R_x.
    mat4_mul(y_rotation_matrix, x_rotation_matrix, model_matrix);
    mat4_mul(z_rotation_matrix, model_matrix, model_matrix);
    mat4_set_translation(model_matrix, position);
}

void mat4_make_camera_model_matrix(const vec3 rotation, const float zoom, mat4 model_matrix)
{
    mat4 x_rotation_matrix;
    mat4 y_rotation_matrix;
    vec3 position = VEC3_FORWARD_INIT;
    mat4_make_x_rotation(rotation[0], x_rotation_matrix);
    mat4_make_y_rotation(rotation[1], y_rotation_matrix);
    mat4_mul(y_rotation_matrix, x_rotation_matrix, model_matrix);
    vec3_scale(position, zoom, position);
    mat4_set_translation(model_matrix, position);
}

void mat4_make_camera_view_matrix(const vec3 position, const vec3 rotation, mat4 view_matrix)
{
    mat4 x_rotation_matrix;
    mat4 y_rotation_matrix;
    mat4_make_x_rotation(-rotation[0], x_rotation_matrix);
    mat4_make_y_rotation(-rotation[1], y_rotation_matrix);
    mat4_mul(y_rotation_matrix, x_rotation_matrix, view_matrix);
    vec3 inverse_position;
    vec3_scale(position, -1.0f, inverse_position);
    mat4_set_translation(view_matrix, inverse_position);
}

void mat4_make_camera_model_and_view_matrix(
    const vec3 rotation, const float zoom, mat4 model_matrix, mat4 view_matrix
){
    mat4_make_camera_model_matrix(rotation, zoom, model_matrix);
    const vec3 position = {model_matrix[3][0], model_matrix[3][1], model_matrix[3][2]};
    mat4_make_camera_view_matrix(position, rotation, view_matrix);
}
/* End of mat4 functions. */

