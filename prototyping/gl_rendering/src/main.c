#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include "shaders.h"
#include "file_io.h"
#include "types.h"
#include "transform.h"

#define DATA_PATH(path) "/home/hayden/repos/g3dm/data/"path

int window_width = 1280;
int window_height = 720;
const char* window_title = "Synthetic 3D";
float window_width_f = 0.0f;
float window_height_f = 0.0f;
float window_height_to_width_ratio = 0.0f;
int randomize_scene = 1;

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

uint32_t create_shader_program(const char* vertex_shader_source, const char* fragment_shader_source)
{
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
    return shader_program;
}

typedef struct
{
    uint32_t vao, vbo, ibo, nbo;
    int num_indices, num_normals, num_vertices;
} gl_mesh_t;

gl_mesh_t mesh_to_gl_mesh(mesh_t* mesh)
{
    uint32_t vao, vbo, ibo, nbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &nbo);
    glBindVertexArray(vao);
    glGenBuffers(1, &ibo);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        sizeof(GL_UNSIGNED_INT) * mesh->num_indices, 
        mesh->indices, 
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        sizeof(GL_FLOAT) * mesh->num_vertices * 3, 
        mesh->vertices, 
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GL_FLOAT) * mesh->num_normals * 3,
        mesh->normals,
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    gl_mesh_t gl_mesh = { 
        .vao = vao, .vbo = vbo, .ibo = ibo, .nbo = nbo,
        .num_indices = mesh->num_indices,
        .num_normals = mesh->num_normals,
        .num_vertices = mesh->num_vertices
    };
    return gl_mesh;
}

typedef struct
{
    uint32_t perspective_matrix_location;
    uint32_t rotation_matrix_location;
    uint32_t object_color_location;
    uint32_t position_offset_location;
    uint32_t ambient_strength_location;
    uint32_t light_position_location;
    uint32_t shader_program;
} mesh_shader_t;

mesh_shader_t shader_program_to_mesh_shader(uint32_t shader_program)
{
    mesh_shader_t mesh_shader;
    mesh_shader.perspective_matrix_location = glGetUniformLocation(shader_program, "perspective_matrix");
    mesh_shader.rotation_matrix_location = glGetUniformLocation(shader_program, "rotation_matrix");
    mesh_shader.object_color_location = glGetUniformLocation(shader_program, "object_color");
    mesh_shader.position_offset_location = glGetUniformLocation(shader_program, "position_offset");
    mesh_shader.ambient_strength_location = glGetUniformLocation(shader_program, "ambient_strength");
    mesh_shader.light_position_location = glGetUniformLocation(shader_program, "light_pos");
    mesh_shader.shader_program = shader_program;
    return mesh_shader;
}

int main()
{
    GLFWwindow* window = init_gl();
    if(!window)
    {
        return -1;
    }

    const char* mesh_path = DATA_PATH("3d_models/sonic/sonic.obj");
    gl_mesh_t gl_mesh;
    mesh_t* mesh = load_obj(mesh_path, 100000, 300000, 100000);
    if(!mesh) { return - 1; }
    gl_mesh = mesh_to_gl_mesh(mesh);
    free(mesh->vertices);
    free(mesh->indices);
    free(mesh->normals);
    free(mesh);
    mesh_shader_t mesh_shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    mat4 perspective_matrix = get_perspective_matrix(60.0f, 0.1f, 1000.0f, aspect_ratio);
    mat4 rotation_matrix = get_y_rotation_matrix(0.0f);
    float mesh_position_offset[3] = {0.0f, -1.5f, -3.0f};
    float object_color[3] = {0.8f, 0.13f, 0.42f};
    float light_position[3] = {1.0f, 1.0f, 0.0f};

    glEnable(GL_DEPTH_TEST);
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(gl_mesh.vao);
        glUseProgram(mesh_shader.shader_program);
        glUniformMatrix4fv(mesh_shader.perspective_matrix_location, 1, GL_FALSE, perspective_matrix.data);
        glUniformMatrix4fv(mesh_shader.rotation_matrix_location, 1, GL_FALSE, rotation_matrix.data);
        glUniform3fv(mesh_shader.position_offset_location, 1, mesh_position_offset);
        glUniform3fv(mesh_shader.object_color_location, 1, object_color);
        glUniform1f(mesh_shader.ambient_strength_location, 0.7f); 
        glUniform3fv(mesh_shader.light_position_location, 1, light_position);
        glDrawElements(GL_TRIANGLES, gl_mesh.num_indices, GL_UNSIGNED_INT, NULL);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
