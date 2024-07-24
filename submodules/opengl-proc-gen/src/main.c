#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include "shaders.h"
#include "file_io.h"
#include "vector_matrix_math.h"
#include "rendering.h"

#define DATA_PATH(path) "/home/hayden/repos/g3dm/data/"path

unsigned int window_width = 512;
unsigned int window_height = 512;
const char* window_title = "3D";
float window_width_f = 0.0f;
float window_height_f = 0.0f;
float window_height_to_width_ratio = 0.0f;
const float fov = 80.0f;

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);    
}

static void window_size_callback(GLFWwindow* window, int width, int height)
{
    window_width = width;
    window_height = height;
    window_width_f = (float)window_width;
    window_height_f = (float)window_height;
    window_height_to_width_ratio = window_height_f / window_width_f;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }
}

GLFWwindow* init_gl(void)
{
    printf("%s\n", shader_vert);
    printf("%s\n", shader_frag);
    if(glfwInit() == GLFW_FALSE)
    {
        printf("Failed to initialize glfw\n");
        return NULL;
    }
    glfwSetErrorCallback(error_callback);

    GLFWwindow* window = glfwCreateWindow(window_width, window_height, window_title, NULL, NULL);
    if(window == NULL)
    {
        printf("Failed to create window\n");
        glfwTerminate();
        return NULL;
    }
    window_width_f = (float)window_width;
    window_height_f = (float)window_height;
    window_height_to_width_ratio = window_height_f / window_width_f;
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    int version = gladLoadGL(glfwGetProcAddress);
    if(version == 0)
    {
        printf("Failed to initialize OpenGL context\n");
        return NULL;
    }
    printf("Loaded OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));
    return window;
}

typedef struct
{
    float min_x_rotation;
    float max_x_rotation;
    float x_rotation_domain;
    float x_rotation_per_step;
    unsigned int x_rotation_steps;
    
    float min_y_rotation;
    float max_y_rotation;
    float y_rotation_domain;
    float y_rotation_per_step;
    unsigned int y_rotation_steps;
    
    unsigned int num_views;
} multi_view_render_params_t;

void init_multi_view_render_params(
    float min_x_rotation, float max_x_rotation, unsigned int x_rotation_steps,
    float min_y_rotation, float max_y_rotation, unsigned int y_rotation_steps,
    multi_view_render_params_t* params
){
    params->min_x_rotation = min_x_rotation;
    params->max_x_rotation = max_x_rotation;
    params->x_rotation_domain = max_x_rotation - min_x_rotation;
    params->x_rotation_per_step = (params->x_rotation_domain / (float)x_rotation_steps) + 1;
    params->x_rotation_steps = x_rotation_steps;

    params->min_y_rotation = min_y_rotation;
    params->max_y_rotation = max_y_rotation;
    params->y_rotation_domain = max_y_rotation - min_y_rotation;
    params->y_rotation_per_step = params->y_rotation_domain / (float)y_rotation_steps;
    params->y_rotation_steps = y_rotation_steps;
    
    params->num_views = x_rotation_steps * y_rotation_steps;
}

void make_multi_view_render_matrices(
    multi_view_render_params_t* params, mat4* model_matrices, mat4* view_matrices
){
    unsigned int matrix_index = 0;
    const float camera_zoom = -4.0f;

    for(unsigned int x = 0; x < params->x_rotation_steps; x++) 
    {
        float x_rotation = params->min_x_rotation + params->x_rotation_per_step * (float)x;
        
        for(unsigned int y = 0; y < params->y_rotation_steps; y++)
        {
            float y_rotation = params->min_y_rotation + params->y_rotation_per_step * (float)y;
            
            vec3 camera_rotation = {x_rotation, y_rotation, 0.0f};
            mat4 model_matrix, view_matrix;
            mat4_make_camera_model_and_view_matrix(
                camera_rotation, camera_zoom, &model_matrix, &view_matrix
            );
            
            memcpy(&model_matrices[matrix_index], &model_matrix, sizeof(mat4));
            memcpy(&view_matrices[matrix_index], &view_matrix, sizeof(mat4));
            ++matrix_index;
        }
    }
}

void multi_view_render(
    const scene_t* scene, camera_t* camera, 
    mesh_shader_t* shader, depth_map_shader_t* depth_shader,
    GLFWwindow* window
){
    const float half_fov_radians = degrees_to_radians(fov/2.0f);
    const float min_x_rotation = -80.0f;
    const float max_x_rotation = 80.0f;
    const float x_rotation_domain = max_x_rotation - min_x_rotation;
    const unsigned int x_rotation_steps = 20;
    const float x_rotation_per_step = x_rotation_domain / (float)x_rotation_steps;
    
    const float min_y_rotation = 0.0f;
    const float max_y_rotation = 360.0f;
    const float y_rotation_domain = max_y_rotation - min_y_rotation;
    const unsigned int y_rotation_steps = 10;
    const float y_rotation_per_step = y_rotation_domain / (float)y_rotation_steps;

    char* base_path = DATA_PATH("multi_view_renders/");
    char save_path[100];
    unsigned int render_index = 0;
    const unsigned int num_views = (x_rotation_steps+1) * y_rotation_steps;
    mat4* transform_matrices = (mat4*)malloc(sizeof(mat4) * num_views);
    unsigned int transform_matrices_offset = 0;
    const float camera_zoom = -4.0f;
    for(unsigned int x = 0; x <= x_rotation_steps; x++)
    {
        const float x_rotation = min_x_rotation + x_rotation_per_step * (float)x;
        for(unsigned int y = 0; y < y_rotation_steps; y++)
        {
            const float y_rotation = min_y_rotation + y_rotation_per_step * (float)y;
            const vec3 camera_rotation = {x_rotation, y_rotation, 0.0f};
            mat4_make_camera_model_and_view_matrix(
                camera_rotation, camera_zoom, camera->model_matrix, camera->view_matrix
            );
            memcpy(
                &transform_matrices[transform_matrices_offset++], 
                camera->model_matrix, 
                sizeof(mat4)
            );
            render_scene(scene, camera, depth_shader, shader, window_width, window_height);
            snprintf(save_path, sizeof(char) * 100, "%s%d%s", base_path, render_index, ".png");
            save_frame_to_png(save_path, window_width, window_height);
            snprintf(save_path, sizeof(char) * 100, "%s%d_depth%s", base_path, render_index, ".png");
            save_depth_to_png(save_path, window_width, window_height);
            glfwSwapBuffers(window);
            render_index++;
        }
    }
    save_multi_view_transforms_json(
        half_fov_radians, 0.0f, num_views, transform_matrices, 
        DATA_PATH("multi_view_renders/transforms.json"), 1
    );
    free(transform_matrices);
}

int main()
{
    GLFWwindow* window = init_gl();
    if(!window)
    {
        return -1;
    }
    scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
    if(!scene)
    {
        printf("Failed to allocate memory for scene\n");
        return -1;
    }
    camera_t* camera = (camera_t*)malloc(sizeof(camera_t));
    if(!camera)
    {
        printf("Failed to allocate memory for camera\n");
        return -1;
    }
    
    obj_t* sonic_obj = load_obj(DATA_PATH("3d_models/sonic/sonic.obj"), 100000, 100000, 100000);
    if(!sonic_obj) 
    {
        printf("Failed to load obj\n");
        return -1; 
    }
    image_t* sonic_texture = load_png(DATA_PATH("3d_models/sonic/sonic.png"));
    if(!sonic_texture) 
    { 
        printf("Failed to load texture\n");
        return -1; 
    }
    
    const vec3 light_rotation = {50.0f, 20.0f, 0.0f};
    init_scene(light_rotation, 0.3f, scene);
    int error = 0;
    unsigned int sonic_mesh_index = 0;
    error = add_mesh_to_scene(scene, sonic_obj, &sonic_mesh_index);
    if(error) { return -1; }
    unsigned int sonic_texture_index = 0;
    error = add_texture_to_scene(scene, sonic_texture, &sonic_texture_index);
    if(error) { return -1; }

    printf("num_gl_meshes: %d\n", scene->num_gl_meshes);
    printf("num_gl_textures: %d\n", scene->num_gl_textures);
    printf("ambient_strength %f\n", scene->light.ambient_strength);
    printf(
        "light_direction %f %f %f\n", 
        scene->light.direction[0], 
        scene->light.direction[1],
        scene->light.direction[2]
    );
    
    mat4 sonic_model_matrix;
    vec3 sonic_position = {0.0f, -1.5f, 0.0f};
    vec3 sonic_rotation = VEC3_ZERO_INIT;
    mat4_make_ordinary_model_matrix(sonic_position, sonic_rotation, sonic_model_matrix);
    error = add_scene_element(scene, sonic_model_matrix, sonic_mesh_index, sonic_texture_index);
    if(error) { return -1; }

    mesh_shader_t shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );
    depth_map_shader_t depth_shader = shader_program_to_depth_map_shader(
        create_shader_program(depth_map_vert, depth_map_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    

    float rot = 0.0f;
    mat4_make_perspective_projection(fov, 1.0f, 20.0f, aspect_ratio, camera->perspective_matrix);
    vec3 camera_rotation = {0.0f, 0.0f, 0.0f};
    const float camera_zoom = -4.0f;
    mat4_make_camera_model_and_view_matrix(
        camera_rotation, camera_zoom, camera->model_matrix, camera->view_matrix
    );
    glEnable(GL_DEPTH_TEST);
    
    multi_view_render_params_t render_params;
    init_multi_view_render_params(-80.0f, 80.0f, 20, 0.0f, 360.0f, 10, &render_params);
    printf("num_views %u\n", render_params.num_views);
    mat4* mv_model_matrices = (mat4*)malloc(sizeof(mat4) * render_params.num_views);
    mat4* mv_view_matrices = (mat4*)malloc(sizeof(mat4) * render_params.num_views);
    make_multi_view_render_matrices(&render_params, mv_model_matrices, mv_view_matrices);

    //multi_view_render(scene, camera, &shader, &depth_shader, window);
    while(!glfwWindowShouldClose(window))
    {
        rot = camera_rotation[1] + 0.4f;
        if(rot > 360.0f) { rot -= 360.0f; }
        else if(rot < 0.0f) { rot += 360.0f; }
        camera_rotation[1] = rot;
        mat4_make_camera_model_and_view_matrix(
            camera_rotation, camera_zoom, camera->model_matrix, camera->view_matrix
        );
        render_scene(scene, camera, &depth_shader, &shader, window_width, window_height);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
