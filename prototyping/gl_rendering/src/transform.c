#include <stdlib.h>
#include <string.h>
#include <math.h>

static double degrees_to_radians(double degrees)
{
    const double pi = 3.14159265358979323846;
    return degrees * (pi / 180.0);
}

float* get_perspective_matrix(float fov, float near_plane, float far_plane, float aspect_ratio)
{
    float tan_half_fov = (float)tan(degrees_to_radians((double)fov) / 2.0);
    float temp_matrix[16] = {
        1.0f / (aspect_ratio * tan_half_fov), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tan_half_fov, 0.0f, 0.0f,
        0.0f, 0.0f, -(far_plane + near_plane) / (far_plane - near_plane), -1.0f,
        0.0f, 0.0f, -2.0f * far_plane * near_plane / (far_plane - near_plane), 0.0f
    };
    size_t matrix_size = sizeof(float) * 16;
    float* matrix = (float*)malloc(matrix_size);
    memcpy(matrix, temp_matrix, matrix_size);
    return matrix;
}
