#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 texture_coord;

uniform mat4 perspective_matrix;
uniform mat4 rotation_matrix;
uniform vec3 position_offset;

out vec3 frag_normal;
out vec3 frag_pos;
out vec2 frag_texture_coord;

void main()
{
    frag_texture_coord = texture_coord;
    frag_normal = vertex_normal;
    vec4 rotated_pos = rotation_matrix * vec4(vertex_pos, 1.0f);
    vec3 shifted_pos = rotated_pos.xyz + position_offset; 
    frag_pos = shifted_pos;
    vec4 projected_pos = perspective_matrix * vec4(shifted_pos, 1.0);
    gl_Position = projected_pos;
}
