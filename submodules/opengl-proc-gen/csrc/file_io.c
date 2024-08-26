#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <png.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "file_io.h"

#define FOUR_SPACE_INDENT(s) "    "s

#define N3DC_OBJ_IMPLEMENTATION
#include <n3dc/n3dc_obj.h>

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
        png, info, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    unsigned char* pixels = (unsigned char*)malloc(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    png_bytep rows[height];
    for(int i = 0; i < height; ++i) 
    {
        rows[height - 1 - i] = pixels + (i * width * 4);
    }

    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(pixels);
    png_destroy_write_struct(&png, &info);
    fclose(file);
}

void save_depth_to_png(const char* filename, unsigned int width, unsigned int height)
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
        png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    unsigned char* depths = (unsigned char*)malloc(width * height);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, depths);

    for(unsigned int i = 0; i < width*height; i++)
    {
        depths[i] = 255 - depths[i];
    }

    png_bytep rows[height];
    for(int i = 0; i < height; ++i) 
    {
        rows[height - 1 - i] = depths + (i * width);
    }

    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(depths);
    png_destroy_write_struct(&png, &info);
    fclose(file);
}

void save_multi_view_transforms_json(
    const float fov_x, const float fov_y,
    const unsigned int num_views, const mat4* transform_matrices,
    const char* file_name, const int with_depth
){
    FILE* file = fopen(file_name, "w");
    if(file == NULL) 
    {
        printf("Error opening file: %s\n", file_name);
        return;
    }

    fprintf(file, "{\n");
    fprintf(file, FOUR_SPACE_INDENT("\"camera_angle_x\": %.7f,\n"), fov_x);
    fprintf(file, FOUR_SPACE_INDENT("\"frames\": [\n"));
    
    const char* column_format = FOUR_SPACE_INDENT(FOUR_SPACE_INDENT(
        FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("[%.7f, %.7f, %.7f, %.7f],\n"))
    ));
    // Same as regular column format but without the trailing comma.
    const char* final_column_format = FOUR_SPACE_INDENT(FOUR_SPACE_INDENT(
        FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("[%.7f, %.7f, %.7f, %.7f]\n"))
    ));
    const char* color_path_format = FOUR_SPACE_INDENT(FOUR_SPACE_INDENT(
        FOUR_SPACE_INDENT("\"file_path\": \"./train/%d.png\",\n")
    ));
    for(unsigned int i = 0; i < num_views; i++) 
    {
        fprintf(file, FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("{\n")));
        fprintf(file, color_path_format, i);
        fprintf(file, FOUR_SPACE_INDENT(FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("\"transform_matrix\": [\n"))));
        const mat4* current_transform = &transform_matrices[i];
        for(unsigned int col = 0; col < 4; col++)
        {
            const float x = (*current_transform)[0][col];
            const float y = (*current_transform)[1][col];
            const float z = (*current_transform)[2][col];
            const float w = (*current_transform)[3][col];
            if(col == 3)
            {
                // Print column without trailing comma if this is the last column.
                fprintf(file, final_column_format, x, y, z, w);
            }
            else
            {
                fprintf(file, column_format, x, y, z, w);
            }
        }
        fprintf(file, FOUR_SPACE_INDENT(FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("]\n"))));
        if(i == num_views - 1)
        {
            // Print closing bracket without trailing comma if this is the last view.
            fprintf(file, FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("}\n")));
        }
        else
        {
            fprintf(file, FOUR_SPACE_INDENT(FOUR_SPACE_INDENT("},\n")));
        }
    }

    fprintf(file, FOUR_SPACE_INDENT("]\n"));
    fprintf(file, "}\n");
    fclose(file);
}
