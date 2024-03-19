#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include "shaders.h"
#include "file_io.h"
#include "mesh.h"
#include "transform.h"

int window_width = 256;
int window_height = 256;
const char* window_title = "Synthetic 3D";
float window_width_f = 0.0f;
float window_height_f = 0.0f;
float window_height_to_width_ratio = 0.0f;
int next_mesh = 0;

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
    else if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        next_mesh = 1;
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
    
    const int num_object_colors = 9;
    const float object_colors[27] = {
        0.95f, 0.22f, 0.1f,
        0.1f, 0.93f, 0.22f,
        0.1f, 0.21f, 0.88f,
        0.99f, 0.62f, 0.33f,
        0.33f, 0.63f, 0.99f,
        0.62f, 0.33f, 0.99f,
        0.33f, 0.58f, 0.27f,
        0.87f, 0.21f, 0.77f,
        0.11f, 0.91f, 0.89f
    };

    const int num_meshes = 27;
    const char mesh_paths[27][100] = {
        "/home/hayden/repos/g3dm/data/monsters/alien.obj",
        "/home/hayden/repos/g3dm/data/monsters/alpaking.obj",
        "/home/hayden/repos/g3dm/data/monsters/armabee.obj",
        "/home/hayden/repos/g3dm/data/monsters/birb.obj",
        "/home/hayden/repos/g3dm/data/monsters/blue_demon.obj",
        "/home/hayden/repos/g3dm/data/monsters/bunny.obj",
        "/home/hayden/repos/g3dm/data/monsters/cactoro.obj",
        "/home/hayden/repos/g3dm/data/monsters/demon.obj",
        "/home/hayden/repos/g3dm/data/monsters/dino.obj",
        "/home/hayden/repos/g3dm/data/monsters/dragon_evolved.obj",
        "/home/hayden/repos/g3dm/data/monsters/dragon.obj",
        "/home/hayden/repos/g3dm/data/monsters/fish.obj",
        "/home/hayden/repos/g3dm/data/monsters/frog.obj",
        "/home/hayden/repos/g3dm/data/monsters/ghost.obj",
        "/home/hayden/repos/g3dm/data/monsters/ghost_skull.obj",
        "/home/hayden/repos/g3dm/data/monsters/glub_evolved.obj",
        "/home/hayden/repos/g3dm/data/monsters/glub.obj",
        "/home/hayden/repos/g3dm/data/monsters/goleling_evolved.obj",
        "/home/hayden/repos/g3dm/data/monsters/goleling.obj",
        "/home/hayden/repos/g3dm/data/monsters/monkroose.obj",
        "/home/hayden/repos/g3dm/data/monsters/mushnub.obj",
        "/home/hayden/repos/g3dm/data/monsters/mushroom_king.obj",
        "/home/hayden/repos/g3dm/data/monsters/orc_skull.obj",
        "/home/hayden/repos/g3dm/data/monsters/pigeon.obj",
        "/home/hayden/repos/g3dm/data/monsters/squidle.obj",
        "/home/hayden/repos/g3dm/data/monsters/tribale.obj",
        "/home/hayden/repos/g3dm/data/monsters/yeti.obj"
    };
    gl_mesh_t gl_meshes[num_meshes]; 
    for(int i = 0; i < num_meshes; i++)
    {
        mesh_t* mesh = load_obj(mesh_paths[i], 100000, 300000, 100000);
        if(!mesh) { return - 1; }
        gl_meshes[i] = mesh_to_gl_mesh(mesh);
        free(mesh->vertices);
        free(mesh->indices);
        free(mesh->normals);
        free(mesh);
    }
    const int num_active_meshes = 3;
    int gl_mesh_indices[3] = {5, 10, 17};
    int object_color_indices[3] = {0, 2, 4};

    mesh_shader_t mesh_shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    mat4 perspective_matrix = get_perspective_matrix(60.0f, 0.1f, 1000.0f, aspect_ratio);
    
    const int num_rotation_matrices = 5;
    mat4 rotation_matrices[5] = {
        get_y_rotation_matrix(0.0f),
        get_y_rotation_matrix(45.0f),
        get_y_rotation_matrix(-45.0f),
        get_y_rotation_matrix(-90.0f),
        get_y_rotation_matrix(90.0f)
    };
    int rotation_indices[3] = {0, 1, 2};    

    float mesh_position_offsets[9] = {
        -2.5f, -2.0f, -6.0f,
        2.5f, -2.0f, -6.0f,
        0.0f, -2.0f, -6.0f
    };

    const int num_environments = 2;
    float ambient_strengths[2] = {0.5f, 0.1f};
    float bg_colors[6] = {
        1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f
    };
    int environment_index = 0;
    
    const int num_light_positions = 3;
    float light_positions[9] = {
        1.0f, 0.2f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 2.0f, -3.0f
    };
    int light_position_index = 0;

    glEnable(GL_DEPTH_TEST);
    while(!glfwWindowShouldClose(window))
    {
        if(next_mesh)
        {
            for(int i = 0; i < num_active_meshes; i++)
            {
                gl_mesh_indices[i]++;
                if(gl_mesh_indices[i] >= num_meshes)
                {
                    gl_mesh_indices[i] = 0;
                }
                object_color_indices[i] = rand() % num_object_colors;
                rotation_indices[i] = rand() % num_rotation_matrices;   
                environment_index = rand() % num_environments;
                light_position_index = rand() % num_light_positions;
            }
            next_mesh = 0;
        }
        const int bg_color_offset = environment_index * 3;
        glClearColor(
            bg_colors[bg_color_offset], 
            bg_colors[bg_color_offset+1], 
            bg_colors[bg_color_offset+2], 
            1.0f
        );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for(int i = 0; i < num_active_meshes; i++)
        {
            gl_mesh_t mesh = gl_meshes[gl_mesh_indices[i]];
            glBindVertexArray(mesh.vao);
            glUseProgram(mesh_shader.shader_program);
            glUniformMatrix4fv(mesh_shader.perspective_matrix_location, 1, GL_FALSE, perspective_matrix.data);
            glUniformMatrix4fv(mesh_shader.rotation_matrix_location, 1, GL_FALSE, rotation_matrices[rotation_indices[i]].data);
            glUniform3fv(mesh_shader.position_offset_location, 1, mesh_position_offsets + (i * 3));
            glUniform3fv(mesh_shader.object_color_location, 1, object_colors + (object_color_indices[i] * 3));
            glUniform1f(mesh_shader.ambient_strength_location, ambient_strengths[environment_index]);
            glUniform3fv(mesh_shader.light_position_location, 1, light_positions + (light_position_index * 3));
            glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, NULL);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    save_frame_to_png("/home/hayden/repos/g3dm/data/gl_output.png", window_width, window_height);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
