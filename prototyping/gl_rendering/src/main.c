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

unsigned int window_width = 1280;
unsigned int window_height = 720;
const char* window_title = "3D";
float window_width_f = 0.0f;
float window_height_f = 0.0f;
float window_height_to_width_ratio = 0.0f;

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

int main()
{
    GLFWwindow* window = init_gl();
    if(!window)
    {
        return -1;
    }

    obj_t* elf_obj = load_obj(DATA_PATH("3d_models/elf/elf.obj"), 100000, 100000, 100000);
    if(!elf_obj) 
    {
        printf("Failed to load obj\n");
        return -1; 
    }
    image_t* texture = get_placeholder_texture(1.0f, 1024, 1024);
    if(!texture) 
    { 
        printf("Failed to load texture\n");
        return -1; 
    }
    obj_t* platform_obj = load_obj(DATA_PATH("3d_models/platform.obj"), 50, 50, 50);
    if(!platform_obj)
    {
        printf("Failed to load platform obj\n");
    }
    
    int error = 0;
    scene_t* scene = init_scene(10.0f, 3.0f, 0.0f, 0.3f);
    unsigned int elf_mesh_index = 0;
    error = add_mesh_to_scene(scene, elf_obj, &elf_mesh_index);
    if(error) { return -1; }
    printf("elf_mesh_index: %d\n", elf_mesh_index);
    unsigned int platform_mesh_index = 0;
    error = add_mesh_to_scene(scene, platform_obj, &platform_mesh_index);
    if(error) { return -1; }
    printf("platform_mesh_index: %d\n", platform_mesh_index);
    unsigned int texture_index = 0;
    error = add_texture_to_scene(scene, texture, &texture_index);
    if(error) { return -1; }
    printf("texture index: %d\n", texture_index);

    printf("num_gl_meshes: %d\n", scene->num_gl_meshes);
    printf("num_gl_textures: %d\n", scene->num_gl_textures);
    printf("ambient_strength %f\n", scene->ambient_strength);
    printf(
        "light_position %f %f %f\n", 
        scene->light_position[0], 
        scene->light_position[1],
        scene->light_position[2]
    );

    mat4 elf_model_matrix = get_model_matrix(0.0f, -1.0f, -3.0f, 0.0f, 0.0f, 0.0f);
    for(int i = 0; i < 16; i++)
    {
        printf("%f ", elf_model_matrix.data[i]);
    }
    printf("\n");
    error = add_scene_element(scene, elf_model_matrix, elf_mesh_index, texture_index);
    if(error) { return -1; }

    mesh_shader_t shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    camera_t camera = {
        .perspective_matrix = get_perspective_matrix(60.0f, 0.1f, 1000.0f, aspect_ratio),
        .view_matrix = get_y_rotation_matrix(0.0)
    };

    glEnable(GL_DEPTH_TEST);
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render_scene(scene, &camera, &shader);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
