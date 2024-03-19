#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;

uniform mat4 perspective_matrix;

out vec3 f_vertex_normal;

void main()
{
    f_vertex_normal = vertex_normal;
    vec3 shifted_pos = vertex_pos - vec3(0.0, 0.0, 3.0); 
    vec4 projected_pos = perspective_matrix * vec4(shifted_pos, 1.0);
    gl_Position = projected_pos;
}
