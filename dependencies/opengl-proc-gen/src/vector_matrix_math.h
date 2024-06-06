#pragma once

typedef float mat4[4][4];
typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];
typedef float quat[4];

#define MAT4_ZERO_INIT \
    { \
        {0.0f, 0.0f, 0.0f, 0.0f}, \
        {0.0f, 0.0f, 0.0f, 0.0f}, \
        {0.0f, 0.0f, 0.0f, 0.0f}, \
        {0.0f, 0.0f, 0.0f, 0.0f} \
    }

#define MAT4_IDENTITY_INIT \
    { \
        {1.0f, 0.0f, 0.0f, 0.0f}, \
        {0.0f, 1.0f, 0.0f, 0.0f}, \
        {0.0f, 0.0f, 1.0f, 0.0f}, \
        {0.0f, 0.0f, 0.0f, 1.0f} \
    }

#define VEC3_ZERO_INIT {0.0f, 0.0f, 0.0f}
#define VEC4_ZERO_INIT {0.0f, 0.0f, 0.0f, 0.0f}
#define VEC3_FORWARD_INIT {0.0f, 0.0f, -1.0f};

float degrees_to_radians(float degrees);

/* Start of vec3 functions. */
void vec3_copy(const vec3 source, vec3 result);
void vec3_scale(const vec3 v, const float s, vec3 result);
void vec3_add(const vec3 a, const vec3 b, vec3 result);
// Uses a mat4 to transform a vec3 by treating the mat4 as a mat3 
// (the 4th row and column are ignored).
void vec3_mat4_mul(const vec3 v, const mat4 m, vec3 result);
/* End of vec3 functions. */

/* Start of vec4 functions. */
float vec4_norm(const vec4 v);
/* End of vec4 functions. */

/* Start of mat4 functions. */
void mat4_copy(const mat4 source, mat4 result);
void mat4_translate(const mat4 m, const vec3 t, mat4 result);
void mat4_set_translation(mat4 m , const vec3 t);
void mat4_mul(const mat4 a, const mat4 b, mat4 result);
void mat4_scale_affine(const mat4 m, const vec3 s, mat4 result);
void mat4_make_identity(mat4 result);
void mat4_make_x_rotation(const float degrees, mat4 result);
void mat4_make_y_rotation(const float degrees, mat4 result);
void mat4_make_z_rotation(const float degrees, mat4 result);
void mat4_make_perspective_projection(
    const float fov, const float near_plane, const float far_plane, 
    const float aspect_ratio, mat4 result
);
/* End of mat4 functions. */

/* Start of quat functions. */
void quat_to_mat4(const quat q, mat4 result);
/* End of quat functions. */
