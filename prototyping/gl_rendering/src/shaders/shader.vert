#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 texture_coord;

uniform mat4 perspective_matrix;
uniform mat4 rotation_matrix;
uniform vec3 position_offset;
uniform vec3 light_pos;

out vec3 frag_normal;
out vec3 frag_pos;
out vec2 frag_texture_coord;
out vec3 frag_light_pos;

void main()
{
    mat4 view_matrix = rotation_matrix;
    view_matrix[3] = vec4(position_offset, 1.0f);
    vec4 view_pos = view_matrix * vec4(vertex_pos, 1.0f);
    gl_Position = perspective_matrix * view_pos;

    frag_light_pos = (rotation_matrix * vec4(light_pos, 1.0f)).xyz;
    frag_pos = view_pos.xyz;
    frag_texture_coord = texture_coord;
    frag_normal = (rotation_matrix * vec4(vertex_normal, 1.0f)).xyz;
}
