#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

image_t* load_png(const char* file_name);
void save_frame_to_png(const char* filename, unsigned int width, unsigned int height);

obj_t* load_obj(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_normals,
    const unsigned int max_indices
);

#ifdef __cplusplus
} // extern "C"
#endif
