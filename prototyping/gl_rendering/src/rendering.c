#include <stdio.h>
#include <stdlib.h>
#include "rendering.h"

uint32_t create_shader_program(
    const char* vertex_shader_source, const char* fragment_shader_source
){
    uint32_t vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, (const char* const*)&vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    
    uint32_t fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, (const char* const*)&fragment_shader_source, NULL);
    glCompileShader(fragment_shader);

    uint32_t shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    printf("shader program %d\n", shader_program);
    return shader_program;
}

mesh_shader_t shader_program_to_mesh_shader(uint32_t shader_program)
{
    mesh_shader_t mesh_shader = {
        .perspective_matrix_location = glGetUniformLocation(shader_program, "perspective_matrix"),
        .view_matrix_location = glGetUniformLocation(shader_program, "view_matrix"),
        .model_matrix_location = glGetUniformLocation(shader_program, "model_matrix"),
        .ambient_strength_location = glGetUniformLocation(shader_program, "ambient_strength"),
        .light_position_location = glGetUniformLocation(shader_program, "light_pos"),
        .texture_location = glGetUniformLocation(shader_program, "texture_sampler"),
        .shader_program = shader_program
    };
    return mesh_shader;
}

image_t* get_placeholder_texture(float value, unsigned int width, unsigned int height)
{
    unsigned int num_scalars = width * height * 4;
    image_t* texture = (image_t*)malloc(sizeof(image_t));
    texture->width = width;
    texture->height = height;
    texture->pixels = (float*)malloc(sizeof(float) * num_scalars);
    for(unsigned int i = 0; i < num_scalars; i++)
    {
        texture->pixels[i] = value;
    }
    return texture;
}

gl_mesh_t obj_to_gl_mesh(obj_t* obj)
{
    uint32_t vao, vbo, nbo, tbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &tbo);
    glBindVertexArray(vao);
    
    const unsigned int num_vertex_scalars = obj->num_vertices * 3;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        sizeof(GL_FLOAT) * num_vertex_scalars, 
        obj->vertices, 
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GL_FLOAT) * num_vertex_scalars,
        obj->normals,
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(1);
   
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GL_FLOAT) * obj->num_vertices * 2,
        obj->texture_coords,
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 2, (void*)0);
    glEnableVertexAttribArray(2);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    gl_mesh_t gl_mesh = { 
        .vao = vao, .vbo = vbo, .nbo = nbo, .tbo = tbo,
        .num_vertices = obj->num_vertices,
        .num_vertex_scalars = num_vertex_scalars
    };
    return gl_mesh;
}

gl_texture_t image_to_gl_texture(image_t* texture)
{
    uint32_t texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        texture->width,
        texture->height,
        0,
        GL_RGBA,
        GL_FLOAT,
        texture->pixels
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    gl_texture_t gl_texture = { .id = texture_id };
    return gl_texture;
}

scene_t* init_scene(float light_x, float light_y, float light_z, float ambient_strength)
{
    scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
    scene->num_gl_meshes = 0;
    scene->num_gl_textures = 0;
    scene->num_elements = 0;
    scene->light_position[0] = light_x;
    scene->light_position[1] = light_y;
    scene->light_position[2] = light_z;
    scene->ambient_strength = ambient_strength;
    return scene;
}

int add_mesh_to_scene(scene_t* scene, obj_t* mesh, unsigned int* mesh_index)
{
    unsigned int new_mesh_count = scene->num_gl_meshes + 1;
    if(new_mesh_count >= MAX_GL_MESHES)
    {
        printf("Could not add mesh to scene because MAX_GL_MESHES was exceeded\n");
        return -1;
    }
    *mesh_index = scene->num_gl_meshes;
    gl_mesh_t gl_mesh = obj_to_gl_mesh(mesh);
    scene->gl_meshes[scene->num_gl_meshes] = gl_mesh;
    scene->num_gl_meshes = new_mesh_count;
    return 0;
}

int add_texture_to_scene(scene_t* scene, image_t* texture, unsigned int* texture_index)
{
    unsigned int new_texture_count = scene->num_gl_textures + 1;
    if(new_texture_count >= MAX_GL_TEXTURES)
    {
        printf("Could not add texture to scene because MAX_GL_TEXTURES was exceeded\n");
        return -1;
    }
    *texture_index = scene->num_gl_textures;
    gl_texture_t gl_texture = image_to_gl_texture(texture);
    scene->gl_textures[scene->num_gl_textures] = gl_texture;
    scene->num_gl_textures = new_texture_count;
    return 0;
}

int add_scene_element(
    scene_t* scene, mat4 model_matrix, 
    const unsigned int mesh_index, const unsigned int texture_index
){
    unsigned int new_element_count = scene->num_elements + 1;
    if(new_element_count >= MAX_SCENE_ELEMENTS)
    {
        printf("Could not add element to scene because MAX_SCENE_ELEMENTS was exceeded\n");
        return -1;
    }
    scene_element_t element = { 
        .model_matrix = model_matrix, 
        .mesh_index = mesh_index, 
        .texture_index = texture_index 
    };
    scene->elements[scene->num_elements] = element;
    scene->num_elements = new_element_count;
    return 0;
}

void render_scene(scene_t* scene, camera_t* camera, mesh_shader_t* shader)
{
    for(unsigned int i = 0; i < scene->num_elements; i++)
    {
        scene_element_t* element = &scene->elements[i];
        gl_mesh_t* mesh = &scene->gl_meshes[element->mesh_index];
        gl_texture_t* texture = &scene->gl_meshes[element->texture_index];
        /*printf("%d\n", mesh->num_vertices);
        printf("%d\n", texture->id);
        printf("%d\n", element->mesh_index);
        printf("%d\n", element->texture_index);
        for(int i = 0; i < 16; i++)
        {
            printf("%f ", camera->view_matrix.data[i]);
        }*/

        glBindVertexArray(mesh->vao);
        glUseProgram(shader->shader_program);
        glUniformMatrix4fv(
            shader->perspective_matrix_location, 1, GL_FALSE, camera->perspective_matrix.data
        );
        glUniformMatrix4fv(
            shader->view_matrix_location, 1, GL_FALSE, camera->view_matrix.data
        );
        glUniformMatrix4fv(
            shader->model_matrix_location, 1, GL_FALSE, element->model_matrix.data
        );
        glUniform1f(shader->ambient_strength_location, scene->ambient_strength); 
        glUniform3fv(shader->light_position_location, 1, scene->light_position);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->id);
        glUniform1i(shader->texture_location, 0);

        glDrawArrays(GL_TRIANGLES, 0, mesh->num_vertices);
    }
}
