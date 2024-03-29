#include "mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

mesh_t* load_obj(
    const char* path, 
    const uint32_t max_vertices, 
    const uint32_t max_indices,
    const uint32_t max_normals
);

void save_frame_to_png(const char* filename, int width, int height);

#ifdef __cplusplus
} // extern "C"
#endif
