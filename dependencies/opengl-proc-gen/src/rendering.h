#pragma once

#include <png.h>
#include <stdint.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "file_io.h"
#include "vector_matrix_math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_GL_MESHES 8
#define MAX_GL_TEXTURES 8
#define MAX_SCENE_ELEMENTS 8

typedef struct
{
    uint32_t vao, vbo, nbo, tbo;
    unsigned int num_vertices;
} gl_mesh_t;

typedef struct
{
    unsigned int mesh_index;
    unsigned int texture_index;
    mat4 model_matrix;
} scene_element_t;

typedef struct
{
    uint32_t depth_map_fbo;
    uint32_t depth_map;
    unsigned int depth_map_width;
    unsigned int depth_map_height;
    mat4 projection_matrix;
    mat4 model_matrix;
    mat4 view_matrix;
    vec3 direction;
    float ambient_strength;
} directional_light_t;

typedef struct
{
    unsigned int num_gl_meshes;
    unsigned int num_gl_textures;
    unsigned int num_elements;
    gl_mesh_t gl_meshes[MAX_GL_MESHES];
    uint32_t gl_textures[MAX_GL_TEXTURES];
    scene_element_t elements[MAX_SCENE_ELEMENTS];
    directional_light_t light;
} scene_t;

typedef struct
{
    uint32_t perspective_matrix_location;
    uint32_t view_matrix_location;
    uint32_t model_matrix_location;
    uint32_t ambient_strength_location;
    uint32_t light_direction_location;
    uint32_t texture_location;
    uint32_t light_projection_matrix_location;
    uint32_t light_view_matrix_location;
    uint32_t depth_map_location;
    uint32_t shader_program;
} mesh_shader_t;

typedef struct
{
    uint32_t model_matrix_location;
    uint32_t light_view_matrix_location;
    uint32_t light_projection_matrix_location;
    uint32_t shader_program;
} depth_map_shader_t;

typedef struct
{
    mat4 perspective_matrix;
    mat4 model_matrix;
    mat4 view_matrix;
} camera_t;

uint32_t create_shader_program(
    const char* vertex_shader_source, const char* fragment_shader_source
);
mesh_shader_t shader_program_to_mesh_shader(uint32_t shader_program);
depth_map_shader_t shader_program_to_depth_map_shader(uint32_t shader_program);

gl_mesh_t obj_to_gl_mesh(obj_t* obj);
uint32_t image_to_gl_texture(image_t* texture);

void init_scene(const vec3 light_rotation, float ambient_strength, scene_t* scene);

int add_mesh_to_scene(scene_t* scene, obj_t* mesh, unsigned int* mesh_index);
int add_texture_to_scene(scene_t* scene, image_t* texture, unsigned int* texture_index);
int add_scene_element(
    scene_t* scene, mat4 model_matrix, 
    const unsigned int mesh_index, const unsigned int texture_index
);

void render_scene(
    scene_t* scene, camera_t* camera, 
    depth_map_shader_t* depth_shader, mesh_shader_t* shader,
    unsigned int viewport_width, unsigned int viewport_height
);

#ifdef __cplusplus
} // extern "C"
#endif
