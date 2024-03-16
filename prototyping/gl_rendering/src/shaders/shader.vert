#version 330 core
layout (location = 0) in vec3 vertex_pos;

out float depth;

void main()
{
    depth = length(vertex_pos);
    gl_Position = vec4(vertex_pos.x, vertex_pos.y, vertex_pos.z, 1.0);
}
