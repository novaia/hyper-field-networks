/* Generated shader header file */
const char* shader_vert_shader = "#version 330 corelayout (location = 0) in vec3 vertex_pos;void main(){    gl_Position = u_vp * vec4(vertex_pos.x, vertex_pos.y, vertex_pos.z, 1.0);}";
const char* shader_frag_shader = "#version 330 corevoid main(){    gl_FragColor = vec4(height, height, height, 1.0f);}";
