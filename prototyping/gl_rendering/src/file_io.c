#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <png.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "types.h"

image_t* load_png(const char* file_name)
{
    FILE* fp = fopen(file_name, "rb");
    if(!fp)
    {
        printf("Error: could not open %s\n", file_name);
        return NULL; 
    }
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png)
    {
        printf("Error: could not create PNG read struct.\n");
        fclose(fp);
        return NULL;
    }
    png_infop info = png_create_info_struct(png);
    if(!info)
    {
        printf("Error: could not create PNG info struct.\n");
        fclose(fp);
        return NULL;
    }
    
    png_init_io(png, fp);
    png_read_info(png, info);
    const unsigned int width = (unsigned int)png_get_image_width(png, info);
    const unsigned int height = (unsigned int)png_get_image_height(png, info);
    const png_byte color_type = png_get_color_type(png, info);
    const png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth == 16)
    {
        png_set_strip_16(png);
    }
    if(color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png);
    }
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if(png_get_valid(png, info, PNG_INFO_tRNS))
    {
        png_set_tRNS_to_alpha(png);
    }
    // These color_type don't have an alpha channel, so fill it with 0xff.
    if(
        color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE
    ){
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }
    if(
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA
    ){
        png_set_gray_to_rgb(png);
    }
    png_read_update_info(png, info);
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++)
    {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }
    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    
    image_t* image = (image_t*)malloc(sizeof(image_t));
    image->width = width;
    image->height = height;
    image->stride = 4;
    image->pixels = (float*)malloc(sizeof(float) * width * height * image->stride);
    for(unsigned int y = 0; y < height; y++)
    {
        unsigned int row_offset = y * width;
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++)
        {
            unsigned int pixel_offset = (row_offset + x) * image->stride;
            png_bytep px = &row[x * image->stride];
            image->pixels[pixel_offset] = px[0]/255.0f;
            image->pixels[pixel_offset + 1] = px[1]/255.0f;
            image->pixels[pixel_offset + 2] = px[2]/255.0f;
            image->pixels[pixel_offset + 3] = px[3]/255.0f;
        }
    }
    free(row_pointers);
    return image;
}

void save_frame_to_png(const char* filename, unsigned int width, unsigned int height)
{
    FILE* file = fopen(filename, "wb");
    if(!file) 
    {
        fprintf(stderr, "Error opening file '%s' for writing\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) 
    {
        fprintf(stderr, "Error creating PNG write struct\n");
        fclose(file);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if(!info) 
    {
        fprintf(stderr, "Error creating PNG info struct\n");
        png_destroy_write_struct(&png, NULL);
        fclose(file);
        return;
    }

    if(setjmp(png_jmpbuf(png))) 
    {
        fprintf(stderr, "Error setting jump buffer for PNG\n");
        png_destroy_write_struct(&png, &info);
        fclose(file);
        return;
    }

    png_init_io(png, file);
    png_set_IHDR(
        png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    unsigned char* pixels = (unsigned char*)malloc(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    png_bytep rows[height];
    for(int i = 0; i < height; ++i) 
    {
        rows[height - 1 - i] = pixels + (i * width * 3);
    }

    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(pixels);
    png_destroy_write_struct(&png, &info);
    fclose(file);
}

static inline size_t get_base_path_length(const char* full_path)
{
    size_t full_path_length = strlen(full_path);
    size_t base_path_length = 0;
    for(size_t i = full_path_length; i > 0; --i)
    {
        if(full_path[i] == '/')
        {
            base_path_length = i+1;
            break;
        }
    }
    return base_path_length;
}

static inline void join_base_path_and_target(
    const char* base_path, const char* target, char* dest, size_t combined_length
){
    snprintf(dest, combined_length, "%s%s", base_path, target);
}

static inline unsigned int min_uint(unsigned int a, unsigned int b) { return (a < b) ? a : b; }

static inline float string_section_to_float(long start, long end, const char* full_string)
{
    unsigned int char_count = min_uint((unsigned int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (float)atof(string_section);
}

static inline unsigned int string_section_to_uint(long start, long end, const char* full_string)
{
    unsigned int char_count = min_uint((unsigned int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (unsigned)atoi(string_section);
}

static int read_text_file(const char* file_path, char** file_chars, long* file_length)
{
    FILE* fp = fopen(file_path, "r");
    if(!fp)
    {
        fprintf(stderr, "Could not open %s\n", file_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    *file_length = ftell(fp);
    rewind(fp);
    *file_chars = (char*)malloc(sizeof(char) * (*file_length + 1));
    if(!file_chars)
    {
        fprintf(stderr, "Could not allocate memory for reading %s\n", file_path);
        fclose(fp);
        return -1;
    }
    size_t read_size = fread(*file_chars, sizeof(char), *file_length, fp);
    (*file_chars)[read_size] = '\0';
    fclose(fp);
}

static material_t* load_mtl(
    const char* mtl_path, const char* base_path, size_t base_path_length
){
    char* file_chars = NULL;
    long file_length = 0;
    int error = read_text_file(mtl_path, &file_chars, &file_length);
    if(error) 
    {
        free(file_chars);
        return NULL; 
    }
    
    unsigned int ignore_current_line = 0;
    unsigned int is_start_of_line = 1;
    unsigned int is_map = 0;
    long map_start = -1;
    long map_end = -1;
    char* texture_path = NULL;
    size_t texture_path_length = 0;

    for(long i = 0; i < file_length; i++)
    {
        const char current_char = file_chars[i];
        if(is_start_of_line)
        {
            // If i == file_length then parsing is finished so the loop can be broken
            // in order to prevent next_char from being out of bounds.
            if(i == file_length) { break; }
            const char next_char = file_chars[i+1];
            if(current_char == 'm' && next_char == 'a')
            {
                is_map = 1;
            }
            else
            {
                ignore_current_line = 1;
            }
            is_start_of_line = 0;
        }
        else if(!ignore_current_line)
        {
            if(is_map)
            {
                if(map_start == -1 && current_char == ' ')
                {
                    map_start = i+1;   
                }
                else if(current_char == '\n')
                {
                    map_end = i;
                    texture_path_length = (size_t)(map_end - map_start);
                    texture_path = (char*)malloc(sizeof(char) * (texture_path_length+1));
                    memcpy(texture_path, &file_chars[map_start], sizeof(char) * texture_path_length);
                    texture_path[texture_path_length] = '\0';            

                    is_map = 0;
                    map_start = -1;
                    map_end = -1;
                }
            }
        }

        if(current_char == '\n')
        {
            is_start_of_line = 1;
            ignore_current_line = 0;
        }
    }

    material_t* mtl_data = (material_t*)malloc(sizeof(material_t));
    mtl_data->texture = NULL;
    if(texture_path)
    {
        size_t full_texture_path_length = base_path_length + texture_path_length + 1;
        char full_texture_path[full_texture_path_length];
        join_base_path_and_target(base_path, texture_path, full_texture_path, full_texture_path_length);
        free(texture_path);
        printf("%s\n", full_texture_path);
        mtl_data->texture = load_png(full_texture_path);
        if(!mtl_data->texture)
        {
            printf("Could not load material's texture\n");
            free(mtl_data);
            free(file_chars);
            return NULL;
        }
    }
    return mtl_data;   
}

static inline int is_valid_vertex_char(const char c)
{
    switch(c)
    {
        case '-': return 1;
        case '.': return 1;
        case '0': return 1;
        case '1': return 1;
        case '2': return 1;
        case '3': return 1;
        case '4': return 1;
        case '5': return 1;
        case '6': return 1;
        case '7': return 1;
        case '8': return 1;
        case '9': return 1;
        default: return 0;
    }
}

static inline int parse_obj_vertex(
    const char* file_chars, const size_t file_chars_length, 
    const size_t vertex_start, size_t* line_end, 
    float* vertex_x, float* vertex_y, float* vertex_z
){
    unsigned int vertex_x_parsed = 0, vertex_y_parsed = 0;
    size_t vertex_x_end = 0, vertex_y_end = 0, vertex_z_end = 0;
    for(size_t i = vertex_start; i < file_chars_length; i++)
    {
        const char current_char = file_chars[i];
        if(current_char == ' ')
        {
            if(!vertex_x_parsed)
            {
                vertex_x_parsed = 1;
                vertex_x_end = i;
            }
            else if(!vertex_y_parsed)
            {
                vertex_y_parsed = 1;
                vertex_y_end = i;
            }
        }
        else if(current_char == '\n')
        {
            if(!vertex_x_parsed)
            {
                printf("Reached end of OBJ vertex line without parsing the x element\n");
                return -1;
            }
            else if(!vertex_y_parsed)
            {
                printf("Reached end of OBJ vertex line without parsing the y element\n");
                return -1;
            }
            vertex_z_end = i;
            *vertex_x = string_section_to_float(vertex_start, vertex_x_end, file_chars);
            *vertex_y = string_section_to_float(vertex_x_end, vertex_y_end, file_chars);
            *vertex_z = string_section_to_float(vertex_y_end, vertex_z_end, file_chars);
            *line_end = vertex_z_end;
            return 0;
        }
        else if(!is_valid_vertex_char(current_char))
        {
            printf("Invalid character encountered when parsing OBJ vertex: \'%c\'\n", current_char);
            return -1;
        }
    }
    printf("Reached end of OBJ file while parsing a vertex\n");
    return -1;
}

static inline int parse_obj_index_group(
    const char* file_chars, const size_t file_chars_length, 
    const size_t index_group_start, size_t* index_group_end,
    unsigned int* vertex_index, unsigned int* texture_index, unsigned int* normal_index
){
    // OBJ index group format: "vertex_index/texture_index/normal_index", where each *_index
    // is 1-based index. Example: 12/2/17.
    // texture_index and normal_index are optional, but we require all of our models to have them
    // and throw an error if they don't.
    unsigned int vertex_index_parsed = 0, texture_index_parsed = 0;
    size_t vertex_index_end = 0, texture_index_end = 0, normal_index_end = 0;
    for(size_t i = index_group_start; i < file_chars_length; i++)
    {
        const char current_char = file_chars[i];
        if(current_char == '/')
        {
            if(!vertex_index_parsed)
            {
                vertex_index_parsed = 1;
                vertex_index_end = i;
            }
            else if(!texture_index_parsed)
            {
                texture_index_parsed = 1;
                texture_index_end = i;
            }
        }
        else if((current_char == ' ' || current_char == '\n') && vertex_index_parsed)
        {
            normal_index_end = i;
            if(!texture_index_parsed)
            {
                printf("Reached end of OBJ index group without parsing the texture index\n");
                return -1;
            }
            // OBJ indices are 1-based so if we get back a 0 index then we know something is wrong.
            // If the index is valid then we subtract 1 to make it 0-based.
            *vertex_index = string_section_to_uint(index_group_start, vertex_index_end, file_chars);
            if(*vertex_index == 0)
            {
                printf("Vertex index of OBJ index group was either missing or invalid\n");
                return -1;
            }
            (*vertex_index)--;
            // Add 1 to the last index end to get the next index's start position.
            // This is so that next index doesn't start with a '/' which would be converted into 0 by atoi().
            *texture_index = string_section_to_uint(vertex_index_end+1, texture_index_end, file_chars);
            if(*texture_index == 0)
            {
                printf("Texture index of OBJ index group was either missing or invalid\n");
                return -1;
            }
            (*texture_index)--;
            *normal_index = string_section_to_uint(texture_index_end+1, normal_index_end, file_chars);
            if(*normal_index == 0)
            {
                printf("Normal index of OBJ index group was either missing or invalid\n");
                return -1;
            }
            (*normal_index)--;
            *index_group_end = normal_index_end;
            return 0;
        }
    }
    printf("Reached end of OBJ file while parsing an index group\n");
    return -1;
}

static inline int parse_obj_face(
    const char* file_chars, const size_t file_chars_length, 
    const size_t face_start, size_t* line_end,
    unsigned int* vertex_index_1, unsigned int* vertex_index_2, unsigned int* vertex_index_3,
    unsigned int* texture_index_1, unsigned int* texture_index_2, unsigned int* texture_index_3,
    unsigned int* normal_index_1, unsigned int* normal_index_2, unsigned int* normal_index_3
){
    unsigned int index_group_1_parsed = 0, index_group_2_parsed = 0;
    size_t current_char_offset = face_start;
    int error = 0;
    while(current_char_offset < file_chars_length)
    {
        size_t index_group_end = current_char_offset;
        if(!index_group_1_parsed)
        {
            error = parse_obj_index_group(
                file_chars, file_chars_length, 
                current_char_offset, &index_group_end,
                vertex_index_1, texture_index_1, normal_index_1
            );
            index_group_1_parsed = 1;
        }
        else if(!index_group_2_parsed)
        {
            error = parse_obj_index_group(
                file_chars, file_chars_length,
                current_char_offset, &index_group_end,
                vertex_index_2, texture_index_2, normal_index_2
            );
            index_group_2_parsed = 1;
        }
        else
        {
            error = parse_obj_index_group(
                file_chars, file_chars_length, 
                current_char_offset, &index_group_end, 
                vertex_index_3, texture_index_3, normal_index_3
            );
            if(file_chars[index_group_end] != '\n')
            {
                printf(
                    "%s %s %s\n",
                    "Parsed 3 OBJ index groups in current face without reaching a newline,",
                    "the OBJ file may have non-triangulated geometry which is not supported",
                    "by this program, or the OBJ file may be corrupted"
                );
                return -1;
            }
            *line_end = index_group_end;
            return 0;
        }

        if(error) { break; }
        current_char_offset = index_group_end;
    }
    // Only print the end of file error if there was no other error that cause the loop to break.
    if(!error)
    {
        printf("Reached end of OBJ file while parsing a face\n");
    }
    return -1;
}

int load_obj_refactor(const char* path, const unsigned int max_vertices, const unsigned int max_indices)
{
    char* file_chars = NULL;
    long file_length = 0;
    const int file_read_error = read_text_file(path, &file_chars, &file_length);
    if(file_read_error) 
    { 
        free(file_chars);
        return -1;
    }
    const size_t file_chars_length = (size_t)file_length;
    size_t current_char_offset = 0;
    
    float* vertices = (float*)malloc(sizeof(float) * max_vertices);
    unsigned int vertex_offset = 0;
    unsigned int parsed_vertices = 0;
    
    const size_t indices_size = sizeof(unsigned int) * max_indices;
    unsigned int* all_indices = (unsigned int*)malloc(indices_size * 3);
    unsigned int* vertex_indices = (unsigned int*)malloc(indices_size);
    unsigned int* texture_indices = (unsigned int*)malloc(indices_size); 
    unsigned int* normal_indices = (unsigned int*)malloc(indices_size); 
    unsigned int parsed_indices = 0;
    unsigned int vertex_index_offset = 0, texture_index_offset = 0, normal_index_offset = 0;

    char last_char = file_chars[current_char_offset++];
    int error = 0;
    while(current_char_offset < file_chars_length)
    {
        const char current_char = file_chars[current_char_offset];
        if(last_char == 'v' && current_char == ' ')
        {
            parsed_vertices++;
            if(parsed_vertices > max_vertices)
            {
                printf("Exceeded maximum number of vertices while parsing OBJ file\n");
                error = 1;
                break;
            }
            size_t line_end = current_char_offset;
            error = parse_obj_vertex(
                file_chars, file_chars_length, current_char_offset, &line_end,
                &vertices[vertex_offset++], &vertices[vertex_offset++], &vertices[vertex_offset++]
            );
            if(error) { break; }
            current_char_offset = line_end;
        }
        else if(last_char == 'f' && current_char == ' ')
        {
            parsed_indices += 3;
            if(parsed_indices > max_indices)
            {
                printf("Exceeded maximum number of indices while parsing OBJ file\n");
                error = 1;
                break;
            }
            size_t line_end = current_char_offset;
            error = parse_obj_face(
                file_chars, file_chars_length, current_char_offset, &line_end, 
                // Index group 1.
                &vertex_indices[vertex_index_offset++], 
                &texture_indices[texture_index_offset++],
                &normal_indices[normal_index_offset++],
                // Index group 2.
                &vertex_indices[vertex_index_offset++], 
                &texture_indices[texture_index_offset++],
                &normal_indices[normal_index_offset++],
                // Index group 3.
                &vertex_indices[vertex_index_offset++], 
                &texture_indices[texture_index_offset++],
                &normal_indices[normal_index_offset++]
            );
            if(error) { break; }
            current_char_offset = line_end;
        }
        
        last_char = file_chars[current_char_offset];
        current_char_offset++;
    }
    if(error)
    {
        free(vertices);
        free(all_indices);
        return -1;
    }
    printf("Parsed %d vertices\n", parsed_vertices);
    printf("Parsed %d indices\n", parsed_indices);
}

mesh_t* load_obj(
    const char* path,
    const unsigned int max_vertices, 
    const unsigned int max_indices,
    const unsigned int max_normals
){
    char* file_chars = NULL;
    long file_length = 0;
    int error = read_text_file(path, &file_chars, &file_length);
    if(error) 
    { 
        free(file_chars);
        return NULL; 
    }

    // Parser line state.    
    unsigned int ignore_current_line = 0;
    unsigned int is_start_of_line = 1;
    unsigned int is_vertex = 0;
    unsigned int is_face = 0;
    unsigned int is_normal = 0;
    unsigned int is_texture_coord = 0;
    unsigned int is_mtl = 0;
    // Vertices.
    unsigned int parsed_vertices = 0;
    unsigned int vertex_offset = 0;
    const size_t vertex_buffer_size = sizeof(uint32_t) * max_vertices * 3;
    float* vertex_buffer = (float*)malloc(vertex_buffer_size);
    long vertex_x_start = -1, vertex_y_start = -1, vertex_z_start = -1, vertex_end = -1;
    // Indices.
    const size_t index_buffer_size = sizeof(unsigned int) * max_indices;
    unsigned int* vertex_index_buffer = (unsigned int*)malloc(index_buffer_size);
    unsigned int* normal_index_buffer = (unsigned int*)malloc(index_buffer_size);
    unsigned int* texture_index_buffer = (unsigned int*)malloc(index_buffer_size);
    int parsed_indices = 0;
    long index_group_start = -1, vertex_index_end = -1, texture_index_end = -1, normal_index_end = -1;
    // Normals.
    unsigned int parsed_normals = 0;
    unsigned int normal_offset = 0;
    const size_t normal_buffer_size = sizeof(float) * max_normals * 3;
    float* normal_buffer = (float*)malloc(normal_buffer_size);
    long normal_x_start = -1, normal_y_start = -1, normal_z_start = -1, normal_end = -1; 
    // Texture coords.
    const unsigned int max_texture_coords = max_normals;
    unsigned int parsed_texture_coords = 0;
    unsigned int texture_coord_offset = 0;
    float* texture_coord_buffer = (float*)malloc(sizeof(float) * max_normals * 2);
    long texture_coord_start = -1, texture_coord_x_end = -1, texture_coord_y_end = -1;
    // MTL paths.
    char* mtl_path = NULL;
    long mtl_path_start = -1, mtl_path_end = -1;
    size_t mtl_path_length = 0;
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
            } else if(current_char == 'f')
            {
                is_face = 1;
            }
            else if(current_char == 'v' && next_char == 'n')
            {
                is_normal = 1;
            }
            else if(current_char == 'm')
            {
                if(mtl_path == NULL)
                {
                    is_mtl = 1;
                }
                else
                {
                    printf("Detected an MTL path but the MTL path has already been set, skipping...");
                    ignore_current_line = 1;
                }
            }
            else if(current_char == 'v' && next_char == 't')
            {
                is_texture_coord = 1;
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
                        free(file_chars);
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
                        free(file_chars);
                        return NULL;
                    }
                    
                    normal_index_end = i;
                    vertex_index_buffer[parsed_indices] = 
                        string_section_to_uint(index_group_start+1, vertex_index_end, file_chars) - 1;
                    texture_index_buffer[parsed_indices] = 
                        string_section_to_uint(vertex_index_end+1, texture_index_end, file_chars) - 1;
                    normal_index_buffer[parsed_indices] =
                        string_section_to_uint(texture_index_end+1, normal_index_end, file_chars) - 1;
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
                        free(file_chars);
                        return NULL;
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
            else if(is_mtl)
            {
                if(mtl_path_start == -1 && current_char == ' ')
                {
                    mtl_path_start = i+1;
                }
                else if(current_char == '\n')
                {

                    mtl_path_end = i;
                    if(mtl_path_start == -1)
                    {
                        printf("MTL path end detected but the start index was not set\n");
                        free(file_chars);
                        return NULL;
                    }
                    mtl_path_length = (size_t)(mtl_path_end - mtl_path_start);
                    mtl_path = (char*)malloc(sizeof(char) * (mtl_path_length+1));
                    memcpy(mtl_path, &file_chars[mtl_path_start], sizeof(char) * mtl_path_length);
                    mtl_path[mtl_path_length] = '\0';            
                    is_mtl = 0;
                    mtl_path_start = -1;
                    mtl_path_end = -1;
                }
            }
            else if(is_texture_coord)
            {
                if(texture_coord_start == -1 && current_char == ' ')
                {
                    texture_coord_start = i+1;
                }
                else if(texture_coord_x_end == -1 && current_char == ' ')
                {
                    texture_coord_x_end = i;
                }
                else if(current_char == '\n')
                {
                    if(parsed_texture_coords >= max_texture_coords)
                    {
                        printf("Exceed maximum number of texture coords in buffer\n");
                        free(file_chars);
                        return NULL;
                    }
                    texture_coord_y_end = i;
                    texture_coord_buffer[texture_coord_offset++] = 
                        string_section_to_float(texture_coord_start, texture_coord_x_end, file_chars);
                    texture_coord_buffer[texture_coord_offset++] = 
                        string_section_to_float(texture_coord_x_end+1, texture_coord_y_end, file_chars);
                    parsed_texture_coords++;
                    
                    is_texture_coord = 0;
                    texture_coord_start = -1;
                    texture_coord_x_end = -1;
                    texture_coord_y_end = -1;
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
    
    printf(
        "Parsed %d vertices, %d normals, %d texture coords, and %d indices\n", 
        parsed_vertices, parsed_normals, parsed_texture_coords, parsed_indices
    );
    
    material_t* material;
    if(mtl_path)
    {
        const size_t base_path_length = get_base_path_length(path);
        char base_path[base_path_length+1];
        memcpy(base_path, path, sizeof(char) * base_path_length);
        base_path[base_path_length] = '\0';

        size_t full_mtl_path_length = base_path_length + mtl_path_length + 1;
        char full_mtl_path[full_mtl_path_length];
        join_base_path_and_target(base_path, mtl_path, full_mtl_path, full_mtl_path_length);
        material = load_mtl(full_mtl_path, base_path, base_path_length);
        if(!material)
        {
            printf("Could not load material\n");
            free(file_chars);
            return NULL;
        }
    }
    const size_t ordered_scalars_size = sizeof(float) * parsed_indices;
    float* ordered_vertices = (float*)malloc(ordered_scalars_size * 3);
    float* ordered_texture_coords = (float*)malloc(ordered_scalars_size * 2);
    float* ordered_normals = (float*)malloc(ordered_scalars_size * 3);
    for(int i = 0; i < parsed_indices; i++)
    {
        const unsigned int vertex_offset = vertex_index_buffer[i] * 3;
        const unsigned int ordered_vertex_offset = i * 3;
        ordered_vertices[ordered_vertex_offset] = vertex_buffer[vertex_offset];
        ordered_vertices[ordered_vertex_offset+1] = vertex_buffer[vertex_offset+1];
        ordered_vertices[ordered_vertex_offset+2] = vertex_buffer[vertex_offset+2];

        const unsigned int texture_coord_offset = texture_index_buffer[i] * 2;
        const unsigned int ordered_texture_coord_offset = i * 2;
        ordered_texture_coords[ordered_texture_coord_offset] = texture_coord_buffer[texture_coord_offset];
        ordered_texture_coords[ordered_texture_coord_offset+1] = texture_coord_buffer[texture_coord_offset+1];
        
        const unsigned int normal_offset = normal_index_buffer[i] * 3;
        const unsigned int ordered_normal_offset = ordered_vertex_offset;
        ordered_normals[ordered_normal_offset] = normal_buffer[normal_offset];
        ordered_normals[ordered_normal_offset+1] = normal_buffer[normal_offset+1];
        ordered_normals[ordered_normal_offset+2] = normal_buffer[normal_offset+2];
    }
    free(vertex_buffer);
    free(texture_coord_buffer);
    free(normal_buffer);
    free(vertex_index_buffer);
    free(texture_index_buffer);
    free(normal_index_buffer);
    free(file_chars);

    mesh_t* mesh = (mesh_t*)malloc(sizeof(mesh_t));
    mesh->num_vertices = parsed_indices;
    mesh->vertices = ordered_vertices;
    mesh->normals = ordered_normals;
    mesh->texture_coords = ordered_texture_coords;
    mesh->material = material;
    return mesh;
}
