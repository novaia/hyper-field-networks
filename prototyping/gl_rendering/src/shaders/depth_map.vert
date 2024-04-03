#version 330 core
layout (location = 0) in vec3 vertex_pos;

uniform mat4 model_matrix;
uniform mat4 light_view_matrix;
uniform mat4 light_projection_matrix;

void main()
{
    gl_Position = light_projection_matrix * light_view_matrix * model_matrix * vertex_pos;
}
