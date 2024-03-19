#version 330 core

in vec3 f_vertex_normal;

void main()
{
    gl_FragColor = vec4(f_vertex_normal, 1.0f);
}
