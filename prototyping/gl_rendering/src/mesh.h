#pragma once

typedef struct 
{ 
    float* vertices; 
    int num_vertices; 
    uint32_t* indices; 
    int num_indices;
    float* normals;
    int num_normals;
} mesh_t;
