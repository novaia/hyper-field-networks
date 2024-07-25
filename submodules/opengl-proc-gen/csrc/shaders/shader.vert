#version 330 core
layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 texture_coord;

uniform mat4 perspective_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;
uniform vec3 light_direction;
uniform mat4 light_projection_matrix;
uniform mat4 light_view_matrix;

out vec3 frag_normal;
out vec3 frag_m_pos;
out vec3 frag_mv_pos;
out vec2 frag_texture_coord;
out vec3 frag_light_direction;

out vec4 frag_light_space_pos;

void main()
{
    vec4 m_pos = model_matrix * vec4(vertex_pos, 1.0f);
    vec4 mv_pos = view_matrix * m_pos;
    vec4 mvp_pos = perspective_matrix * mv_pos;
    //vec4 mv_pos = light_view_matrix * m_pos;
    //vec4 mvp_pos = light_projection_matrix * mv_pos;
    gl_Position = mvp_pos;
    
    mat3 normal_matrix = mat3(model_matrix);
    frag_normal = normal_matrix * vertex_normal;
    frag_light_direction = light_direction;
    frag_m_pos = m_pos.xyz / m_pos.w;
    frag_mv_pos = mv_pos.xyz / mv_pos.w;
    frag_texture_coord = texture_coord;

    frag_light_space_pos = light_projection_matrix * (light_view_matrix * m_pos);
}
