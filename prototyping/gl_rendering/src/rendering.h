#pragma once

#include <stdint.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "file_io.h"
#include "matrices.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_GL_MESHES 8
#define MAX_GL_TEXTURES 8
#define MAX_SCENE_ELEMENTS 8

typedef struct
{
    uint32_t vao, vbo, ibo, nbo, tbo, texture_id;
    unsigned int num_vertices, num_vertex_scalars;
} gl_mesh_t;

typedef struct
{
    uint32_t id;
} gl_texture_t;

typedef struct
{
    unsigned int mesh_index;
    unsigned int texture_index;
    mat4 model_matrix;
} scene_element_t;

typedef struct
{
    unsigned int num_gl_meshes;
    unsigned int num_gl_textures;
    unsigned int num_elements;
    gl_mesh_t gl_meshes[MAX_GL_MESHES];
    gl_texture_t gl_textures[MAX_GL_TEXTURES];
    scene_element_t elements[MAX_SCENE_ELEMENTS];
    float light_position[3];
    float ambient_strength;
} scene_t;

typedef struct
{
    uint32_t perspective_matrix_location;
    uint32_t view_matrix_location;
    uint32_t model_matrix_location;
    uint32_t ambient_strength_location;
    uint32_t light_position_location;
    uint32_t texture_location;
    uint32_t shader_program;
} mesh_shader_t;

typedef struct
{
    mat4 perspective_matrix;
    mat4 view_matrix;
} camera_t;

uint32_t create_shader_program(
    const char* vertex_shader_source, const char* fragment_shader_source
);
mesh_shader_t shader_program_to_mesh_shader(uint32_t shader_program);

image_t* get_placeholder_texture(float value, unsigned int width, unsigned int height);

gl_mesh_t obj_to_gl_mesh(obj_t* obj);
gl_texture_t image_to_gl_texture(image_t* texture);

scene_t* init_scene(float light_x, float light_y, float light_z, float ambient_strength);
int add_mesh_to_scene(scene_t* scene, obj_t* mesh, unsigned int* mesh_index);
int add_texture_to_scene(scene_t* scene, image_t* texture, unsigned int* texture_index);
int add_scene_element(
    scene_t* scene, mat4 model_matrix, 
    const unsigned int mesh_index, const unsigned int texture_index
);

void render_scene(scene_t* scene, camera_t* camera, mesh_shader_t* shader);

#ifdef __cplusplus
} // extern "C"
#endif
