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

void make_multi_view_render_matrices(
    float min_zoom, float max_zoom,
    unsigned int num_views,
    float min_x_rotation, float max_x_rotation,
    float min_y_rotation, float max_y_rotation,
    mat4* model_matrices, mat4* view_matrices
){
    for(unsigned int i = 0; i < num_views; i++) 
    {
        float x_rotation = min_x_rotation + ((float)rand() / RAND_MAX) * (max_x_rotation - min_x_rotation);
        float y_rotation = min_y_rotation + ((float)rand() / RAND_MAX) * (max_y_rotation - min_y_rotation);
        
        float camera_zoom = min_zoom + ((float)rand() / RAND_MAX) * (max_zoom - min_zoom);

        vec3 camera_rotation = {x_rotation, y_rotation, 0.0f};
        mat4 model_matrix, view_matrix;
        mat4_make_camera_model_and_view_matrix(
            camera_rotation, camera_zoom, &model_matrix, &view_matrix
        );
        
        memcpy(&model_matrices[i], &model_matrix, sizeof(mat4));
        memcpy(&view_matrices[i], &view_matrix, sizeof(mat4));
    }
}

void multi_view_render(
    const scene_t* scene, camera_t* camera, 
    mesh_shader_t* shader, depth_map_shader_t* depth_shader,
    const unsigned int num_views,
    mat4* model_matrices, mat4* view_matrices,
    GLFWwindow* window
){
    const float half_fov_radians = degrees_to_radians(fov/2.0f);
    char* base_path = DATA_PATH("multi_view_renders/");
    char save_path[100];

    for(unsigned int view_index = 0; view_index < num_views; view_index++)
    {
        memcpy(camera->model_matrix, &model_matrices[view_index], sizeof(mat4));
        memcpy(camera->view_matrix, &view_matrices[view_index], sizeof(mat4));

        render_scene(scene, camera, depth_shader, shader, window_width, window_height);

        snprintf(save_path, sizeof(char) * 100, "%s%d%s", base_path, view_index, ".png");
        save_frame_to_png(save_path, window_width, window_height);

        snprintf(save_path, sizeof(char) * 100, "%s%d_depth%s", base_path, view_index, ".png");
        save_depth_to_png(save_path, window_width, window_height);

        glfwSwapBuffers(window);
    }

    save_multi_view_transforms_json(
        half_fov_radians, 0.0f, num_views, model_matrices, 
        DATA_PATH("multi_view_renders/transforms.json"), 1
    );
}

int main()
{
    srand(1729); // RNG seed.

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
    
    n3dc_obj_t* sonic_obj = n3dc_obj_load(DATA_PATH("3d_models/sonic/sonic.obj"), 100000, 100000, 100000);
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

    n3dc_obj_t* camera_obj = n3dc_obj_load(DATA_PATH("3d_models/camera.obj"), 100000, 100000, 100000);
    if(!sonic_obj) 
    {
        printf("Failed to load obj\n");
        return -1; 
    }
    image_t* camera_texture = load_png(DATA_PATH("3d_models/white_texture.png"));
    if(!sonic_texture) 
    { 
        printf("Failed to load texture\n");
        return -1; 
    }
    
    const vec3 light_rotation = {50.0f, 180.0f, 0.0f};
    init_scene(light_rotation, 0.3f, scene);
    int error = 0;
    unsigned int sonic_mesh_index = 0;
    error = add_mesh_to_scene(scene, sonic_obj, &sonic_mesh_index);
    if(error) { return -1; }
    unsigned int sonic_texture_index = 0;
    error = add_texture_to_scene(scene, sonic_texture, &sonic_texture_index);
    if(error) { return -1; }

    unsigned int camera_mesh_index = 0;
    error = add_mesh_to_scene(scene, camera_obj, &camera_mesh_index);
    if(error) { return -1; }
    unsigned int camera_texture_index = 0;
    error = add_texture_to_scene(scene, camera_texture, &camera_texture_index);
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
    mat4_make_perspective_projection(fov, 1.0, 20.0f, aspect_ratio, camera->perspective_matrix);
    vec3 camera_rotation = {0.0f, 0.0f, 0.0f};
    const float camera_zoom = -4.0f;
    mat4_make_camera_model_and_view_matrix(
        camera_rotation, camera_zoom, camera->model_matrix, camera->view_matrix
    );
    glEnable(GL_DEPTH_TEST);
    
    const unsigned int num_views = 200;
    const float min_zoom = -6.0f, max_zoom = -4.0f;
    const float min_x_rotation = -88.0f, max_x_rotation = 88.0f;
    const float min_y_rotation = 0.0f, max_y_rotation = 360.0f;
    mat4* mv_model_matrices = (mat4*)malloc(sizeof(mat4) * num_views);
    mat4* mv_view_matrices = (mat4*)malloc(sizeof(mat4) * num_views);
    make_multi_view_render_matrices(
        min_zoom, max_zoom, num_views,
        min_x_rotation, max_x_rotation,
        min_y_rotation, max_y_rotation,
        mv_model_matrices, mv_view_matrices
    );

    int debug_camera_poses = 0;
    if(debug_camera_poses)
    {
        for(unsigned int i = 0; i < num_views; ++i)
        {
            error = add_scene_element(
                scene, mv_model_matrices[i], camera_mesh_index, camera_texture_index
            );
            if(error) { return -1; }
        }
    }

    multi_view_render(
        scene, camera, &shader, &depth_shader,
        num_views, mv_model_matrices, mv_view_matrices, window
    );

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
