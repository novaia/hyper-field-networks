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
    const int width = png_get_image_width(png, info);
    const int height = png_get_image_height(png, info);
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
    for(int y = 0; y < height; y++)
    {
        int row_offset = y * width;
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++)
        {
            int pixel_offset = (row_offset + x) * image->stride;
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

void save_frame_to_png(const char* filename, int width, int height)
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

static inline int min_int(int a, int b) { return (a < b) ? a : b; }

static inline float string_section_to_float(long start, long end, char* full_string)
{
    int char_count = min_int((int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (float)atof(string_section);
}

static inline int string_section_to_int(long start, long end, char* full_string)
{
    int char_count = min_int((int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (int)atoi(string_section);
}

material_t* load_mtl(const char* path)
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
    
    int ignore_current_line = 0;
    int is_start_of_line = 1;
    int is_map = 0;
    long map_start = -1;
    long map_end = -1;
    char* texture_path = NULL;
    long texture_path_length = 0;

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
                    texture_path_length = map_end - map_start;
                    texture_path = (char*)malloc(sizeof(char) * (texture_path_length+1));
                    long texture_char_offset = map_start;
                    for(long k = 0; k < texture_path_length; k++)
                    {
                        texture_path[k] = file_chars[texture_char_offset++];
                    }
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
        uint32_t mtl_path_length = (uint32_t)strlen(path);
        uint32_t mtl_root_path_length = 0;
        for(uint32_t i = mtl_path_length; i > 0; --i)
        {
            if(path[i] == '/') 
            { 
                mtl_root_path_length = i+1;
                break;
            }
        }
        char mtl_root_path[mtl_root_path_length+1];
        memcpy(mtl_root_path, path, sizeof(char) * mtl_root_path_length);
        mtl_root_path[mtl_root_path_length] = '\0';
        uint32_t full_texture_path_length = mtl_root_path_length + texture_path_length + 1;
        char full_texture_path[full_texture_path_length];
        snprintf(full_texture_path, full_texture_path_length, "%s%s", mtl_root_path, texture_path);
        printf("%s\n", full_texture_path);
        mtl_data->texture = load_png(full_texture_path);
        if(!mtl_data->texture)
        {
            printf("Could not load material's texture\n");
            free(mtl_data);
            return NULL;
        }
    }
    return mtl_data;   
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
    int is_texture_coord = 0;
    int is_mtl = 0;
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
    uint32_t* texture_index_buffer = (uint32_t*)malloc(index_buffer_size);
    int parsed_indices = 0;
    long index_group_start = -1, vertex_index_end = -1, texture_index_end = -1, normal_index_end = -1;
    // Normals.
    int parsed_normals = 0;
    int normal_offset = 0;
    const size_t normal_buffer_size = sizeof(float) * max_normals * 3;
    float* normal_buffer = (float*)malloc(normal_buffer_size);
    long normal_x_start = -1, normal_y_start = -1, normal_z_start = -1, normal_end = -1; 
    // Texture coords.
    const uint32_t max_texture_coords = max_normals;
    int parsed_texture_coords = 0;
    int texture_coord_offset = 0;
    float* texture_coord_buffer = (float*)malloc(sizeof(float) * max_normals * 2);
    long texture_coord_start = -1, texture_coord_x_end = -1, texture_coord_y_end = -1;
    // MTL paths.
    char* mtl_path = NULL;
    long mtl_path_start = -1, mtl_path_end = -1, mtl_path_length = 0;

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
                    texture_index_buffer[parsed_indices] = 
                        (uint32_t)string_section_to_int(vertex_index_end+1, texture_index_end, file_chars) - 1;
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
                        return NULL;
                    }
                    mtl_path_length = mtl_path_end - mtl_path_start;
                    mtl_path = (char*)malloc(sizeof(char) * (mtl_path_length+1));
                    long mtl_char_offset = mtl_path_start;
                    for(long k = 0; k < mtl_path_length; k++)
                    {
                        mtl_path[k] = file_chars[mtl_char_offset++];
                    }
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
        uint32_t obj_path_length = (uint32_t)strlen(path);
        uint32_t obj_root_path_length = 0;
        for(uint32_t i = obj_path_length; i > 0; --i)
        {
            if(path[i] == '/') 
            { 
                obj_root_path_length = i+1;
                break;
            }
        }
        char obj_root_path[obj_root_path_length+1];
        memcpy(obj_root_path, path, sizeof(char) * obj_root_path_length);
        obj_root_path[obj_root_path_length] = '\0';
        uint32_t full_mtl_path_length = obj_root_path_length + mtl_path_length + 1;
        char full_mtl_path[full_mtl_path_length];
        snprintf(full_mtl_path, full_mtl_path_length, "%s%s", obj_root_path, mtl_path);
        printf("%s\n", full_mtl_path);
        material = load_mtl(full_mtl_path);
        if(!material)
        {
            printf("Could not load material\n");
            return NULL;
        }
    }
    const size_t ordered_scalars_size = sizeof(float) * parsed_indices;
    float* ordered_vertices = (float*)malloc(ordered_scalars_size * 3);
    float* ordered_texture_coords = (float*)malloc(ordered_scalars_size * 2);
    float* ordered_normals = (float*)malloc(ordered_scalars_size * 3);
    for(int i = 0; i < parsed_indices; i++)
    {
        const uint32_t vertex_offset = vertex_index_buffer[i] * 3;
        const uint32_t ordered_vertex_offset = i * 3;
        ordered_vertices[ordered_vertex_offset] = vertex_buffer[vertex_offset];
        ordered_vertices[ordered_vertex_offset+1] = vertex_buffer[vertex_offset+1];
        ordered_vertices[ordered_vertex_offset+2] = vertex_buffer[vertex_offset+2];

        const uint32_t texture_coord_offset = texture_index_buffer[i] * 2;
        const uint32_t ordered_texture_coord_offset = i * 2;
        ordered_texture_coords[ordered_texture_coord_offset] = texture_coord_buffer[texture_coord_offset];
        ordered_texture_coords[ordered_texture_coord_offset+1] = texture_coord_buffer[texture_coord_offset+1];
        
        const uint32_t normal_offset = normal_index_buffer[i] * 3;
        const uint32_t ordered_normal_offset = ordered_vertex_offset;
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

    mesh_t* mesh = (mesh_t*)malloc(sizeof(mesh_t));
    mesh->num_vertices = parsed_indices;
    mesh->vertices = ordered_vertices;
    mesh->normals = ordered_normals;
    mesh->texture_coords = ordered_texture_coords;
    mesh->material = material;
    return mesh;
}


