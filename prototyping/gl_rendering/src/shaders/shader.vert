#version 330 core
layout (location = 0) in vec3 vertex_pos;

uniform mat4 perspective_matrix;

out vec3 vertex_color;

void main()
{
    vec3 shifted_pos = vertex_pos - vec3(0.0, 0.0, 3.0); 
    vec4 projected_pos = perspective_matrix * vec4(shifted_pos, 1.0);
    float depth = length(vec3(projected_pos.xyz)) / 5.0;
    vertex_color = vec3(depth, projected_pos.x, projected_pos.y);
    gl_Position = projected_pos;
}
