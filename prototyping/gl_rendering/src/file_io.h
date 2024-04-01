#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

int load_obj_refactor(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_normals,
    const unsigned int max_indices
);

mesh_t* load_obj(
    const char* path, 
    const unsigned int max_vertices, 
    const unsigned int max_indices,
    const unsigned int max_normals
);

void save_frame_to_png(const char* filename, unsigned int width, unsigned int height);

#ifdef __cplusplus
} // extern "C"
#endif
