#version 330 core

in float depth;

void main()
{
    gl_FragColor = vec4(depth, depth, depth, 1.0f);
}
