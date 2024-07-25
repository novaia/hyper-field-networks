#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 texture_coord;

uniform mat4 model_matrix;
uniform mat4 light_view_matrix;
uniform mat4 light_projection_matrix;

void main()
{
    gl_Position = light_projection_matrix * (
        light_view_matrix * (model_matrix * vec4(vertex_pos, 1.0f))
    );
}
