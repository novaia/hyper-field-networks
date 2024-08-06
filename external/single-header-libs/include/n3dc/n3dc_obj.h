#ifndef N3DC_OBJ_H
#define N3DC_OBJ_H

#ifdef __cplusplus
extern "C" {
#endif

#define N3DC_OBJ_VERSION_MAJOR 0
#define N3DC_OBJ_VERSION_MINOR 1
#define N3DC_OBJ_VERSION_PATCH 0

typedef struct 
{ 
    unsigned int num_vertices;
    float* vertices; 
    float* normals;
    float* texture_coords;
} obj_t;

obj_t* load_obj(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_normals,
    const unsigned int max_indices
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // N3DC_OBJ_H

#ifdef N3DC_OBJ_IMPLEMENTATION


static unsigned int min_uint(unsigned int a, unsigned int b) { return (a < b) ? a : b; }

static float string_section_to_float(long start, long end, const char* full_string)
{
    unsigned int char_count = min_uint((unsigned int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (float)atof(string_section);
}

static unsigned int string_section_to_uint(long start, long end, const char* full_string)
{
    unsigned int char_count = min_uint((unsigned int)(end - start), 10);
    char string_section[char_count+1];
    string_section[char_count] = '\0';
    memcpy(string_section, &full_string[start], sizeof(char) * char_count);
    return (unsigned)atoi(string_section);
}

static int is_valid_vec_char(const char c)
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

// Vertices and normals are both vec3's so they can be parsed with the same function.
static int parse_obj_vec3(
    const char* file_chars, const size_t file_chars_length, 
    const size_t vec3_start, size_t* line_end, 
    float* x, float* y, float* z
){
    unsigned int x_parsed = 0, y_parsed = 0;
    size_t x_end = 0, y_end = 0, z_end = 0;
    for(size_t i = vec3_start; i < file_chars_length; i++)
    {
        const char current_char = file_chars[i];
        if(current_char == ' ')
        {
            if(!x_parsed)
            {
                x_parsed = 1;
                x_end = i;
            }
            else if(!y_parsed)
            {
                y_parsed = 1;
                y_end = i;
            }
        }
        else if(current_char == '\n')
        {
            if(!x_parsed)
            {
                printf("Reached end of OBJ vertex/normal line without parsing the x element\n");
                return -1;
            }
            else if(!y_parsed)
            {
                printf("Reached end of OBJ vertex/normal line without parsing the y element\n");
                return -1;
            }
            z_end = i;
            *x = string_section_to_float(vec3_start, x_end, file_chars);
            *y = string_section_to_float(x_end, y_end, file_chars);
            *z = string_section_to_float(y_end, z_end, file_chars);
            *line_end = z_end;
            return 0;
        }
        else if(!is_valid_vec_char(current_char))
        {
            printf("Invalid character encountered when parsing OBJ vertex/normal: \'%c\'\n", current_char);
            return -1;
        }
    }
    printf("Reached end of OBJ file while parsing a vertex/normal\n");
    return -1;
}

static int parse_obj_vec2(
    const char* file_chars, const size_t file_chars_length, 
    const size_t vec2_start, size_t* line_end, 
    float* x, float* y
){
    unsigned int x_parsed = 0;
    size_t x_end = 0, y_end = 0;
    for(size_t i = vec2_start; i < file_chars_length; i++)
    {
        const char current_char = file_chars[i];
        if(current_char == ' ')
        {
            x_parsed = 1;
            x_end = i;
        }
        else if(current_char == '\n')
        {
            if(!x_parsed)
            {
                printf("Reached end of OBJ texture coord line without parsing the x element\n");
                return -1;
            }
            y_end = i;
            *x = string_section_to_float(vec2_start, x_end, file_chars);
            *y = string_section_to_float(x_end, y_end, file_chars);
            *line_end = y_end;
            return 0;
        }
        else if(!is_valid_vec_char(current_char))
        {
            printf("Invalid character encountered while parsing OBJ texture coord: \'%c\'\n", current_char);
            return -1;
        }
    }
    printf("Reached end of OBJ file while parsing a texture coord\n");
    return -1;
}

static int parse_obj_index_group(
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

static int parse_obj_face(
    const char* file_chars, const size_t file_chars_length, 
    const size_t face_start, size_t* line_end,
    unsigned int* vertex_index_1, unsigned int* texture_index_1, unsigned int* normal_index_1,
    unsigned int* vertex_index_2, unsigned int* texture_index_2, unsigned int* normal_index_2,
    unsigned int* vertex_index_3, unsigned int* texture_index_3, unsigned int* normal_index_3
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

static int seek_end_of_line(
    const char* file_chars, const size_t file_chars_length, 
    const size_t start_offset, size_t* line_end 
){
    for(size_t i = start_offset; start_offset < file_chars_length; i++)
    {
        if(file_chars[i] == '\n')
        {
            *line_end = i;
            return 0;
        }
    }
    printf("Reached end of file while seeking end of line\n");
    return -1;
}

static int n3dc_obj_read_text_file(const char* file_path, char** file_chars, long* file_length)
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
    return 0;
}

obj_t* load_obj(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_normals,
    const unsigned int max_indices
){
    char* file_chars = NULL;
    long file_length = 0;
    const int file_read_error = n3dc_obj_read_text_file(path, &file_chars, &file_length);
    if(file_read_error) 
    { 
        free(file_chars);
        return NULL;
    }
    const size_t file_chars_length = (size_t)file_length;
    size_t current_char_offset = 1;
    
    float* vertices = (float*)malloc(sizeof(float) * max_vertices * 3);
    unsigned int vertex_offset = 0;
    unsigned int parsed_vertices = 0;
    
    float* texture_coords = (float*)malloc(sizeof(float) * max_indices * 2);
    unsigned int texture_coord_offset = 0;
    unsigned int parsed_texture_coords = 0;

    float* normals = (float*)malloc(sizeof(float) * max_normals * 3);
    unsigned int normal_offset = 0;
    unsigned int parsed_normals = 0;

    const size_t indices_size = sizeof(unsigned int) * max_indices;
    unsigned int* vertex_indices = (unsigned int*)malloc(indices_size);
    unsigned int* texture_indices = (unsigned int*)malloc(indices_size); 
    unsigned int* normal_indices = (unsigned int*)malloc(indices_size); 
    unsigned int parsed_indices = 0;
    unsigned int index_offset = 0;
    
    int error = 0;
    while(current_char_offset < file_chars_length)
    {
        const char last_char = file_chars[current_char_offset - 1];
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
            // Increment by 1 to skip over space space character.
            current_char_offset++;
            size_t line_end = current_char_offset;
            const unsigned int x_offset = vertex_offset++;
            const unsigned int y_offset = vertex_offset++;
            const unsigned int z_offset = vertex_offset++;
            error = parse_obj_vec3(
                file_chars, file_chars_length, current_char_offset, &line_end,
                &vertices[x_offset], &vertices[y_offset], &vertices[z_offset]
            );
            if(error) { break; }
            current_char_offset = line_end + 2;
        }
        else if(last_char == 'v' && current_char == 't')
        {
            parsed_texture_coords++;
            if(parsed_texture_coords > max_indices)
            {
                printf("Exceed maximum number of texture coords (max_indices) while parsing OBJ file\n");
                error = -1;
                break;
            }
            // Increment by 2 to skip over t and space characters.
            current_char_offset += 2;
            size_t line_end = current_char_offset;
            const unsigned int x_offset = texture_coord_offset++;
            const unsigned int y_offset = texture_coord_offset++;
            error = parse_obj_vec2(
                file_chars, file_chars_length, current_char_offset, &line_end,
                &texture_coords[x_offset], &texture_coords[y_offset]
            );
            if(error) { break; }
            current_char_offset = line_end + 2;

        }
        else if(last_char == 'v' && current_char == 'n')
        {
            parsed_normals++;
            if(parsed_normals > max_normals)
            {
                printf("Exceeded maximum number of normals while parsing OBJ file\n");
                error = 1;
                break;
            }
            // Incremement by 2 to skip over n and space characters.
            current_char_offset += 2;
            size_t line_end = current_char_offset;
            const unsigned int x_offset = normal_offset++;
            const unsigned int y_offset = normal_offset++;
            const unsigned int z_offset = normal_offset++;
            error = parse_obj_vec3(
                file_chars, file_chars_length, current_char_offset, &line_end,
                &normals[x_offset], &normals[y_offset],&normals[z_offset]
            );
            if(error) { break; }
            current_char_offset = line_end + 2;
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
            const unsigned int group_1_offset = index_offset++;
            const unsigned int group_2_offset = index_offset++;
            const unsigned int group_3_offset = index_offset++;
            error = parse_obj_face(
                file_chars, file_chars_length, current_char_offset, &line_end, 
                // Index group 1.
                &vertex_indices[group_1_offset],
                &texture_indices[group_1_offset],
                &normal_indices[group_1_offset],
                // Index group 2.
                &vertex_indices[group_2_offset],
                &texture_indices[group_2_offset],
                &normal_indices[group_2_offset],
                // Index group 3.
                &vertex_indices[group_3_offset],
                &texture_indices[group_3_offset],
                &normal_indices[group_3_offset]
            );
            if(error) { break; }
            current_char_offset = line_end + 2;
        }
        else 
        {
            // Skip to start of next line.
            size_t line_end = current_char_offset;
            error = seek_end_of_line(
                file_chars, file_chars_length, current_char_offset, &line_end
            );
            if(error) { break; }
            current_char_offset = line_end + 2;
        }
    }
    if(error)
    {
        free(vertices);
        free(texture_coords);
        free(normals);
        free(vertex_indices);
        free(texture_indices);
        free(normal_indices);
        free(file_chars);
        return NULL;
    }

    const size_t ordered_scalars_size = sizeof(float) * parsed_indices;
    float* ordered_vertices = (float*)malloc(ordered_scalars_size * 3);
    float* ordered_texture_coords = (float*)malloc(ordered_scalars_size * 2);
    float* ordered_normals = (float*)malloc(ordered_scalars_size * 3);
    for(int i = 0; i < parsed_indices; i++)
    {
        vertex_offset = vertex_indices[i] * 3;
        const unsigned int ordered_vertex_offset = i * 3;
        ordered_vertices[ordered_vertex_offset] = vertices[vertex_offset];
        ordered_vertices[ordered_vertex_offset+1] = vertices[vertex_offset+1];
        ordered_vertices[ordered_vertex_offset+2] = vertices[vertex_offset+2];

        texture_coord_offset = texture_indices[i] * 2;
        const unsigned int ordered_texture_coord_offset = i * 2;
        ordered_texture_coords[ordered_texture_coord_offset] = texture_coords[texture_coord_offset];
        ordered_texture_coords[ordered_texture_coord_offset+1] = texture_coords[texture_coord_offset+1];
        
        normal_offset = normal_indices[i] * 3;
        const unsigned int ordered_normal_offset = ordered_vertex_offset;
        ordered_normals[ordered_normal_offset] = normals[normal_offset];
        ordered_normals[ordered_normal_offset+1] = normals[normal_offset+1];
        ordered_normals[ordered_normal_offset+2] = normals[normal_offset+2];
    }

    free(vertices);
    free(texture_coords);
    free(normals);
    free(vertex_indices);
    free(texture_indices);
    free(normal_indices);
    free(file_chars);

    obj_t* obj = (obj_t*)malloc(sizeof(obj_t));
    obj->num_vertices = parsed_indices;
    obj->vertices = ordered_vertices;
    obj->normals = ordered_normals;
    obj->texture_coords = ordered_texture_coords;
    return obj;
}

#endif // N3DC_OBJ_IMPLEMENTATION
