#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
        .light_direction_location= glGetUniformLocation(shader_program, "light_direction"),
        .texture_location = glGetUniformLocation(shader_program, "texture_sampler"),
        .depth_map_location = glGetUniformLocation(shader_program, "depth_map_sampler"),
        .light_view_matrix_location = glGetUniformLocation(shader_program, "light_view_matrix"),
        .light_projection_matrix_location = glGetUniformLocation(
            shader_program, "light_projection_matrix"
        ),
        .shader_program = shader_program
    };
    return mesh_shader;
}

depth_map_shader_t shader_program_to_depth_map_shader(uint32_t shader_program)
{
    depth_map_shader_t depth_shader = {
        .model_matrix_location = glGetUniformLocation(shader_program, "model_matrix"),
        .light_view_matrix_location = glGetUniformLocation(shader_program, "light_view_matrix"),
        .light_projection_matrix_location = glGetUniformLocation(
            shader_program, "light_projection_matrix"
        ),
        .shader_program = shader_program
    };
    return depth_shader;
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
        .num_vertices = obj->num_vertices
    };
    return gl_mesh;
}

uint32_t image_to_gl_texture(image_t* texture)
{
    uint32_t gl_texture;
    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
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
    return gl_texture;
}

static void setup_directional_light(
    directional_light_t* light,
    const unsigned int depth_map_width, const unsigned int depth_map_height,
    const float light_rotation_x, const float light_rotation_y, const float light_rotation_z,
    const float ambient_strength
){
    light->depth_map_width = depth_map_width;
    light->depth_map_height = depth_map_height;
    light->ambient_strength = ambient_strength;
    glGenFramebuffers(1, &light->depth_map_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, light->depth_map_fbo);

    glGenTextures(1, &light->depth_map);
    glBindTexture(GL_TEXTURE_2D, light->depth_map);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
        light->depth_map_width, light->depth_map_height, 
        0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color); 

    glBindFramebuffer(GL_FRAMEBUFFER, light->depth_map_fbo);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, light->depth_map, 0
    );
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) 
    {
        printf("Framebuffer was not completed\n");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    light->projection_matrix = get_orthogonal_matrix(
        -5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 20.0f
    );
    light->view_matrix = get_lookat_matrix_from_rotation(
        light_rotation_x, light_rotation_y, light_rotation_z, 10.0f
    );
    // Extract light direction from forward component of look-at matrix.
    for(unsigned int i = 0; i < 3; i++)
    {
        light->direction[i] = light->view_matrix.data[4+i];
    }
}

scene_t* init_scene(
    float light_rotation_x, float light_rotation_y, float light_rotation_z, 
    float ambient_strength
){
    scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
    scene->num_gl_meshes = 0;
    scene->num_gl_textures = 0;
    scene->num_elements = 0;
    setup_directional_light(
        &scene->light, 4096, 4096, 
        light_rotation_x, light_rotation_y, light_rotation_z, 
        ambient_strength
    );
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
    uint32_t gl_texture = image_to_gl_texture(texture);
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

void render_scene(
    scene_t* scene, camera_t* camera, 
    depth_map_shader_t* depth_shader, mesh_shader_t* shader,
    unsigned int viewport_width, unsigned int viewport_height
){
    directional_light_t light = scene->light;
    
    // First pass, render depth map.
    glViewport(0, 0, light.depth_map_width, light.depth_map_height);
    glBindFramebuffer(GL_FRAMEBUFFER, light.depth_map_fbo);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(depth_shader->shader_program);
    for(unsigned int i = 0; i < scene->num_elements; i++)
    {
        scene_element_t element = scene->elements[i];
        gl_mesh_t mesh = scene->gl_meshes[element.mesh_index];
        glBindVertexArray(mesh.vao);
        glUniformMatrix4fv(
            depth_shader->model_matrix_location, 1, GL_FALSE, element.model_matrix.data
        );
        glUniformMatrix4fv(
            depth_shader->light_view_matrix_location, 1, GL_FALSE, light.view_matrix.data
        );
        glUniformMatrix4fv(
            depth_shader->light_projection_matrix_location, 
            1, GL_FALSE, light.projection_matrix.data
        );

        glDrawArrays(GL_TRIANGLES, 0, mesh.num_vertices);
    }
    
    // Second pass, render normally.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_width, viewport_height);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shader->shader_program);
    for(unsigned int i = 0; i < scene->num_elements; i++)
    {
        scene_element_t element = scene->elements[i];
        gl_mesh_t mesh = scene->gl_meshes[element.mesh_index];
        uint32_t texture = scene->gl_textures[element.texture_index];
        
        glBindVertexArray(mesh.vao);
        glUniformMatrix4fv(
            shader->perspective_matrix_location, 1, GL_FALSE, camera->perspective_matrix.data
        );
        glUniformMatrix4fv(
            shader->view_matrix_location, 1, GL_FALSE, camera->view_matrix.data
        );
        glUniformMatrix4fv(
            shader->model_matrix_location, 1, GL_FALSE, element.model_matrix.data
        );
        glUniformMatrix4fv(
            shader->light_projection_matrix_location, 
            1, GL_FALSE, light.projection_matrix.data
        );
        glUniformMatrix4fv(
            shader->light_view_matrix_location, 
            1, GL_FALSE, light.view_matrix.data
        );

        glUniform1f(shader->ambient_strength_location, light.ambient_strength); 
        glUniform3fv(shader->light_direction_location, 1, light.direction);

        glUniform1i(shader->texture_location, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glActiveTexture(GL_TEXTURE0 + 1);
        glUniform1i(shader->depth_map_location, 1);
        glBindTexture(GL_TEXTURE_2D, light.depth_map);
        
        glDrawArrays(GL_TRIANGLES, 0, mesh.num_vertices);
    }
}
