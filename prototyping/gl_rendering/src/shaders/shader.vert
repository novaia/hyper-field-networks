#version 330 core
layout (location = 0) in vec3 vertex_pos;

void main()
{
    gl_Position = u_vp * vec4(vertex_pos.x, vertex_pos.y, vertex_pos.z, 1.0);
}
