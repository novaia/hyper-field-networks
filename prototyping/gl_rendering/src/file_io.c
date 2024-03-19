#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "mesh.h"

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

static inline int string_section_to_int(long start, long end, char* full_string)
{
    int char_count = min_int((int)(end - start), 10);
    char string_section[char_count+1];
    for(int i = 0; i < char_count; i++)
    {
        string_section[i] = full_string[start + (long)i];
    }
    string_section[char_count] = '\0';
    return (int)atoi(string_section);
}

mesh_t* load_obj(
    const char* path, 
    const uint32_t max_vertices, 
    const uint32_t max_indices,
    const uint32_t max_normals
){
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
    
    // Parser line state.    
    int ignore_current_line = 0;
    int is_start_of_line = 1;
    int is_vertex = 0;
    int is_face = 0;
    int is_normal = 0;
    // Vertices.
    int parsed_vertices = 0;
    int vertex_offset = 0;
    const size_t vertex_buffer_size = sizeof(uint32_t) * max_vertices * 3;
    float* vertex_buffer = (float*)malloc(vertex_buffer_size);
    long vertex_x_start = -1, vertex_y_start = -1, vertex_z_start = -1, vertex_end = -1;
    // Indices.
    const size_t index_buffer_size = sizeof(uint32_t) * max_indices;
    uint32_t* vertex_index_buffer = (uint32_t*)malloc(index_buffer_size);
    uint32_t* normal_index_buffer = (uint32_t*)malloc(index_buffer_size);
    int parsed_indices = 0;
    long index_group_start = -1, vertex_index_end = -1, texture_index_end = -1, normal_index_end = -1;
    // Normals.
    int parsed_normals = 0;
    int normal_offset = 0;
    const size_t normal_buffer_size = sizeof(float) * max_normals * 3;
    float* normal_buffer = (float*)malloc(normal_buffer_size);
    long normal_x_start = -1, normal_y_start = -1, normal_z_start = -1, normal_end = -1; 
    for(long i = 0; i < file_length; i++)
    {
        const char current_char = file_chars[i];
        if(is_start_of_line)
        {
            // If i == file_length then parsing is finished so the loop can be broken
            // in order to prevent next_char from being out of bounds.
            if(i == file_length) { break; }
            
            const char next_char = file_chars[i+1];
            if(current_char == '#' || current_char == 'o')
            {
                // Ignore the rest of this line.
                ignore_current_line = 1;
            }
            else if(current_char == 'v' && next_char == ' ')
            {
                is_vertex = 1;
            }
            else if(current_char == 'f')
            {
                is_face = 1;
            }
            else if(current_char == 'v' && next_char == 'n')
            {
                is_normal = 1;
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
                    if(parsed_vertices >= max_vertices)
                    {
                        printf("Exceeded maximum number of vertices in buffer\n");
                        return NULL;
                    }
                    
                    vertex_end = i;
                    vertex_buffer[vertex_offset++] = 
                        string_section_to_float(vertex_x_start, vertex_y_start, file_chars);
                    vertex_buffer[vertex_offset++] = 
                        string_section_to_float(vertex_y_start, vertex_z_start, file_chars);
                    vertex_buffer[vertex_offset++] = 
                        string_section_to_float(vertex_z_start, vertex_end, file_chars);
                    parsed_vertices++;
                    
                    // Reset state for next line.
                    vertex_x_start = vertex_y_start = vertex_z_start = vertex_end = -1;
                    is_vertex = 0;
                }
            }
            else if(is_face)
            {
                if(current_char == ' ' && index_group_start == -1)
                {
                    index_group_start = i;
                }
                else if(current_char == '/')
                {
                    if(vertex_index_end == -1) { vertex_index_end = i; }
                    else if(texture_index_end == -1) { texture_index_end = i; }
                }
                else if(current_char == ' ' || current_char == '\n')
                {
                    if(parsed_indices >= max_indices)
                    {
                        printf("Exceeded maximum number of indices in buffer\n");
                        return NULL;
                    }

                    normal_index_end = i;
                    vertex_index_buffer[parsed_indices] = 
                        (uint32_t)string_section_to_int(index_group_start+1, vertex_index_end, file_chars) - 1;
                    normal_index_buffer[parsed_indices] =
                        (uint32_t)string_section_to_int(texture_index_end+1, normal_index_end, file_chars) - 1;
                    parsed_indices++;

                    if(current_char == '\n') 
                    { 
                        // Reset state for next line.
                        is_face = 0; 
                        index_group_start = -1;
                    }
                    else
                    {
                        // Start the next index group.
                        index_group_start = i;
                    }
                    vertex_index_end = texture_index_end = normal_index_end = -1;
                }
            }
            else if(is_normal)
            {
                if(current_char == ' ')
                {
                    if(normal_x_start == -1) { normal_x_start = i; }
                    else if(normal_y_start == -1) { normal_y_start = i; }
                    else if(normal_z_start == -1) { normal_z_start = i; }
                }
                else if(current_char == '\n')
                {
                    if(parsed_normals >= max_normals)
                    {
                        printf("Exceeded maximum number of normals in buffer\n");
                    }

                    normal_end = i;
                    normal_buffer[normal_offset++] = 
                        string_section_to_float(normal_x_start, normal_y_start, file_chars);
                    normal_buffer[normal_offset++] = 
                        string_section_to_float(normal_y_start, normal_z_start, file_chars);
                    normal_buffer[normal_offset++] = 
                        string_section_to_float(normal_z_start, normal_end, file_chars);
                    parsed_normals++;
                    
                    // Reset state for next line.
                    normal_x_start = normal_y_start = normal_z_start = normal_end = -1;
                    is_normal = 0;
                }               
            }
        }
        
        if(current_char == '\n')
        {
            // Reset state for next line.
            ignore_current_line = 0;
            is_start_of_line = 1;
        }
    }

    const size_t parsed_vertices_size = sizeof(float) * parsed_vertices * 3;
    // Resolve the normal indices to get an ordered normal buffer.
    // There should be one normal in the ordered_normal_buffer for each vertex in the vertex_buffer.
    float* ordered_normal_buffer = (float*)malloc(parsed_vertices_size);
    for(int i = 0; i < parsed_indices; i++)
    {
        const int ordered_normal_offset = vertex_index_buffer[i] * 3;
        const int unordered_normal_offset = normal_index_buffer[i] * 3;
        ordered_normal_buffer[ordered_normal_offset] = normal_buffer[unordered_normal_offset];
        ordered_normal_buffer[ordered_normal_offset + 1] = normal_buffer[unordered_normal_offset + 1];
        ordered_normal_buffer[ordered_normal_offset + 2] = normal_buffer[unordered_normal_offset + 2];
    }

    mesh_t* mesh = (mesh_t*)malloc(sizeof(mesh_t));
    mesh->vertices = (float*)malloc(parsed_vertices_size);
    mesh->num_vertices = parsed_vertices;
    memcpy(mesh->vertices, vertex_buffer, parsed_vertices_size);

    const size_t parsed_indices_size = sizeof(uint32_t) * parsed_indices;
    mesh->indices = (uint32_t*)malloc(sizeof(uint32_t) * parsed_indices);
    mesh->num_indices = parsed_indices;
    memcpy(mesh->indices, vertex_index_buffer, parsed_indices_size);

    mesh->normals = (float*)malloc(parsed_vertices_size);
    mesh->num_normals = parsed_vertices;
    memcpy(mesh->normals, ordered_normal_buffer, parsed_vertices_size);
    
    free(vertex_buffer);
    free(vertex_index_buffer);
    free(normal_index_buffer);
    free(normal_buffer);
    free(ordered_normal_buffer);
    return mesh;
}

void save_frame_to_png(const char* filename, int width, int height)
{
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file '%s' for writing\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error creating PNG write struct\n");
        fclose(file);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error creating PNG info struct\n");
        png_destroy_write_struct(&png, NULL);
        fclose(file);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error setting jump buffer for PNG\n");
        png_destroy_write_struct(&png, &info);
        fclose(file);
        return;
    }

    png_init_io(png, file);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    unsigned char* pixels = (unsigned char*)malloc(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    png_bytep rows[height];
    for (int i = 0; i < height; ++i) {
        rows[height - 1 - i] = pixels + (i * width * 3);
    }

    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(pixels);
    png_destroy_write_struct(&png, &info);
    fclose(file);
}
