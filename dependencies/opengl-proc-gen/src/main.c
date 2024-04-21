#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include "shaders.h"
#include "file_io.h"
#include "matrices.h"
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

void multi_view_render(
    const scene_t* scene, camera_t* camera, 
    mesh_shader_t* shader, depth_map_shader_t* depth_shader,
    GLFWwindow* window
){
    const float min_x_rotation = -80.0f;
    const float max_x_rotation = 80.0f;
    const float x_rotation_domain = max_x_rotation - min_x_rotation;
    const unsigned int x_rotation_steps = 5;
    const float x_rotation_per_step = x_rotation_domain / (float)x_rotation_steps;
    
    const float min_y_rotation = 0.0f;
    const float max_y_rotation = 360.0f;
    const float y_rotation_domain = max_y_rotation - min_y_rotation;
    const unsigned int y_rotation_steps = 12;
    const float y_rotation_per_step = y_rotation_domain / (float)y_rotation_steps;

    char* base_path = DATA_PATH("multi_view_renders/");
    char save_path[100];
    unsigned int render_index = 0;
    const unsigned int num_views = (x_rotation_steps+1) * y_rotation_steps;
    mat4* transform_matrices = (mat4*)malloc(sizeof(mat4) * num_views);
    unsigned int transform_matrices_offset = 0;
    for(unsigned int x = 0; x <= x_rotation_steps; x++)
    {
        const float x_rotation = min_x_rotation + x_rotation_per_step * (float)x;
        for(unsigned int y = 0; y < y_rotation_steps; y++)
        {
            const float y_rotation = min_y_rotation + y_rotation_per_step * (float)y;
            camera->view_matrix = get_lookat_matrix_from_rotation(x_rotation, y_rotation, 0.0f, 5.0f);
            const mat4 transform_matrix = get_model_matrix_from_rotation(-x_rotation, -y_rotation, 0.0f, -5.0f);
            transform_matrices[transform_matrices_offset++] = transform_matrix;
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
        (float)degrees_to_radians((double)(fov/2.0f)), 0.0f, num_views, transform_matrices, 
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
    obj_t* elf_obj = load_obj(DATA_PATH("3d_models/elf/elf.obj"), 100000, 100000, 100000);
    if(!elf_obj) 
    {
        printf("Failed to load obj\n");
        return -1; 
    }
    obj_t* plane_obj = load_obj(DATA_PATH("3d_models/platform.obj"), 50, 50, 50);
    if(!plane_obj)
    {
        printf("Failed to load plane obj\n");
    }
    image_t* white_texture = load_png(DATA_PATH("3d_models/white_texture.png"));
    if(!white_texture) 
    { 
        printf("Failed to load texture\n");
        return -1; 
    }
    image_t* orange_texture = load_png(DATA_PATH("3d_models/orange_texture.png"));
    if(!orange_texture) 
    { 
        printf("Failed to load texture\n");
        return -1; 
    }

    
    int error = 0;
    scene_t* scene = init_scene(50.0f, 20.0f, 0.0f, 0.3f);
    unsigned int sonic_mesh_index = 0;
    error = add_mesh_to_scene(scene, sonic_obj, &sonic_mesh_index);
    if(error) { return -1; }
    unsigned int elf_mesh_index = 0;
    error = add_mesh_to_scene(scene, elf_obj, &elf_mesh_index);
    if(error) { return -1; }
    unsigned int plane_mesh_index = 0;
    error = add_mesh_to_scene(scene, plane_obj, &plane_mesh_index);
    if(error) { return -1; }
    unsigned int white_texture_index = 0;
    error = add_texture_to_scene(scene, white_texture, &white_texture_index);
    if(error) { return -1; }
    unsigned int orange_texture_index = 0;
    error = add_texture_to_scene(scene, orange_texture, &orange_texture_index);
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
    
    mat4 elf_model_matrix = get_model_matrix(2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    printf("\n");
    error = add_scene_element(scene, elf_model_matrix, elf_mesh_index, white_texture_index);
    if(error) { return -1; }

    mat4 elf_model_matrix_2 = get_model_matrix(-2.0f, -1.0f, 0.0f, 0.0f, 80.0f, 0.0f);
    error = add_scene_element(scene, elf_model_matrix_2, elf_mesh_index, white_texture_index);
    if(error) { return -1; }
    
    mat4 sonic_model_matrix = get_model_matrix(0.0f, -1.5f, 0.0f, 0.0f, 0.0f, 0.0f);
    error = add_scene_element(scene, sonic_model_matrix, sonic_mesh_index, sonic_texture_index);
    if(error) { return -1; }

    /*mat4 plane_model_matrix = get_model_matrix(0.0f, -1.5f, 0.0f, 0.0f, 0.0f, 0.0f);
    error = add_scene_element(
        scene, plane_model_matrix, plane_mesh_index, orange_texture_index
    );
    if(error) { return -1; }*/
    
    mesh_shader_t shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );
    depth_map_shader_t depth_shader = shader_program_to_depth_map_shader(
        create_shader_program(depth_map_vert, depth_map_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    camera_t camera = {
        .perspective_matrix = get_perspective_matrix(fov, 1.0f, 20.0f, aspect_ratio),
        .view_matrix = get_y_rotation_matrix(0.0)
    };

    float rot = 0.0f;
    glEnable(GL_DEPTH_TEST);
    multi_view_render(scene, &camera, &shader, &depth_shader, window);
    while(!glfwWindowShouldClose(window))
    {
        rot += 0.4f;
        if(rot > 360.0f) { rot -= 360.0f; }
        else if(rot < 0.0f) { rot += 360.0f; }
        camera.view_matrix = get_lookat_matrix_from_rotation(10.0f, rot, 0.0f, 4.0f);
        render_scene(scene, &camera, &depth_shader, &shader, window_width, window_height);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
