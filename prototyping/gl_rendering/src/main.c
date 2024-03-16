#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include <math.h>
#include "shaders.h"

int window_width = 1280;
int window_height = 720;
const char* window_title = "Synthetic 3D";
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

typedef struct { float x, y, z; } vec3_t;
typedef struct { float* vertices; int num_vertices; } mesh_t;

static inline int min_int(int a, int b) { return (a < b) ? a : b; }

static inline float string_section_to_float(long start, long end, char* full_string)
{
    int char_count = min_int((int)(end - start), 10);
    char string_section[char_count+1];
    for(int i = 0; i < char_count; i++)
    {
        string_section[i] = full_string[start + (long)i];
    }
    string_section[char_count] = '\0';
    return (float)atof(string_section);
}

mesh_t* load_obj(const char* path)
{
    FILE* fp = fopen(path, "r");
    if(!fp)
    {
        fprintf(stderr, "Could not open %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long file_length = ftell(fp);
    rewind(fp);
    char* file_chars = malloc(sizeof(char) * (file_length + 1));
    if(!file_chars)
    {
        fprintf(stderr, "Could not allocate memory for reading %s\n", path);
        fclose(fp);
        return NULL;
    }
    size_t read_size = fread(file_chars, sizeof(char), file_length, fp);
    file_chars[read_size] = '\0';
    fclose(fp);
    
    const int max_vertices = 1000;
    int parsed_vertices = 0;
    vec3_t vertex_buffer[max_vertices];
    int ignore_current_line = 0;
    int is_start_of_line = 1;
    int is_vertex = 0;
    long vertex_x_start, vertex_y_start, vertex_z_start, vertex_end = -1;
    for(long i = 0; i < file_length; i++)
    {
        char current_char = file_chars[i];
        if(is_start_of_line)
        {
            if(current_char == '#' || current_char == 'o')
            {
                // Ignore the rest of this line.
                ignore_current_line = 1;
            }
            else if(current_char == 'v' && file_chars[i+1] == ' ')
            {
                is_vertex = 1;
            }
            is_start_of_line = 0;
        }
        else if(!ignore_current_line)
        {
            if(is_vertex)
            {
                if(current_char == ' ')
                {
                    if(vertex_x_start == -1) { vertex_x_start = i; }
                    else if(vertex_y_start == -1) { vertex_y_start = i; }
                    else if(vertex_z_start == -1) { vertex_z_start = i; }
                }
                else if(current_char == '\n')
                {
                    if(parsed_vertices < max_vertices)
                    {
                        vertex_end = i;
                        vertex_buffer[parsed_vertices].x = string_section_to_float(vertex_x_start, vertex_y_start, file_chars);
                        vertex_buffer[parsed_vertices].y = string_section_to_float(vertex_y_start, vertex_z_start, file_chars);
                        vertex_buffer[parsed_vertices].z = string_section_to_float(vertex_z_start, vertex_end, file_chars);
                        parsed_vertices++;
                    }
                    else
                    {
                        printf("Exceed maximum number of vertices in buffer\n");
                        return NULL;
                    }
                }
            }
        }

        if(current_char == '\n')
        {
            // Reset flags for next line.
            ignore_current_line = 0;
            is_start_of_line = 1;
            is_vertex = 0;
            vertex_x_start = -1;
            vertex_y_start = -1;
            vertex_z_start = -1;
            vertex_end = -1;
        }
    }
    
    mesh_t* mesh = (mesh_t*)malloc(sizeof(mesh_t));
    mesh->vertices = (float*)malloc(sizeof(float) * parsed_vertices * 3);
    mesh->num_vertices = parsed_vertices;
    for(int i = 0; i < parsed_vertices; i++)
    {
        const int vertex_offset = i * 3;
        mesh->vertices[vertex_offset] = vertex_buffer[i].x;
        mesh->vertices[vertex_offset + 1] = vertex_buffer[i].y;
        mesh->vertices[vertex_offset + 2] = vertex_buffer[i].z;
        //vec3_t current_vertex = vertex_buffer[i];
        //printf("Vertex %d: %f %f %f\n", i, current_vertex.x, current_vertex.y, current_vertex.z);
    }
    return mesh;
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

static double degrees_to_radians(double degrees)
{
    const double pi = 3.14159265358979323846;
    return degrees * (pi / 180.0);
}

float* get_perspective_matrix(float fov, float near_plane, float far_plane)
{
    float tan_half_fov = (float)tan(degrees_to_radians((double)fov) / 2.0);
    float aspect_ratio = window_width_f / window_height_f;
    float temp_matrix[16] = {
        1.0f / (aspect_ratio * tan_half_fov), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tan_half_fov, 0.0f, 0.0f,
        0.0f, 0.0f, -(far_plane + near_plane) / (far_plane - near_plane), -1.0f,
        0.0f, 0.0f, -2.0f * far_plane * near_plane / (far_plane - near_plane), 0.0f
    };
    size_t matrix_size = sizeof(float) * 16;
    float* matrix = (float*)malloc(matrix_size);
    memcpy(matrix, temp_matrix, matrix_size);
    return matrix;
}

int main()
{
    GLFWwindow* window = init_gl();
    if(!window)
    {
        return -1;
    }
    
    mesh_t* mesh = load_obj("/home/hayden/repos/g3dm/data/monkey.obj");
    float* perspective_matrix = get_perspective_matrix(60.0f, 0.1f, 1000.0f);
    uint32_t shader_program = create_shader_program(shader_vert, shader_frag);
    const uint32_t perspective_matrix_location = glGetUniformLocation(shader_program, "perspective_matrix");
    
    uint32_t vao, vbo, ibo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        sizeof(GL_FLOAT) * mesh->num_vertices * 3, 
        mesh->vertices, 
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(vao);
        glUseProgram(shader_program);
        glUniformMatrix4fv(perspective_matrix_location, 1, GL_FALSE, perspective_matrix);
        glPointSize(5.0f);
        glDrawArrays(GL_POINTS, 0, mesh->num_vertices);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
