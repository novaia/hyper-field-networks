#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;

uniform mat4 perspective_matrix;
uniform mat4 rotation_matrix;
uniform vec3 object_color;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_object_color;

void main()
{
    frag_normal = vertex_normal;
    vec4 rotated_pos = rotation_matrix * vec4(vertex_pos, 1.0f);
    vec3 shifted_pos = rotated_pos.xyz - vec3(0.0, 2.0, 3.0); 
    frag_pos = shifted_pos;
    frag_object_color = object_color;
    vec4 projected_pos = perspective_matrix * vec4(shifted_pos, 1.0);
    gl_Position = projected_pos;
}
