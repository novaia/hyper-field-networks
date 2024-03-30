gl_mesh_t mesh_to_gl_mesh(mesh_t* mesh)
{
    uint32_t vao, vbo, nbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &nbo);
    glBindVertexArray(vao);
    
    const uint32_t num_vertex_scalars = mesh->num_vertices * 3;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        sizeof(GL_FLOAT) * num_vertex_scalars, 
        mesh->vertices, 
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(GL_FLOAT) * num_vertex_scalars,
        mesh->normals,
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, (void*)0);
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    gl_mesh_t gl_mesh = { 
        .vao = vao, .vbo = vbo, .nbo = nbo,
        .num_vertices = mesh->num_vertices,
        .num_vertex_scalars = num_vertex_scalars
    };
    return gl_mesh;
}

int main()
{
    GLFWwindow* window = init_gl();
    if(!window)
    {
        return -1;
    }

    const char* mesh_path = DATA_PATH("3d_models/sonic/sonic.obj");
    gl_mesh_t gl_mesh;
    mesh_t* mesh = load_obj(mesh_path, 100000, 300000, 100000);
    if(!mesh) { return - 1; }
    return 0;
    gl_mesh = mesh_to_gl_mesh(mesh);
    free(mesh->vertices);
    free(mesh->normals);
    free(mesh->texture_coords);
    free(mesh->material);
    free(mesh);
    mesh_shader_t mesh_shader = shader_program_to_mesh_shader(
        create_shader_program(shader_vert, shader_frag)
    );

    float aspect_ratio = window_width_f / window_height_f;
    mat4 perspective_matrix = get_perspective_matrix(60.0f, 0.1f, 1000.0f, aspect_ratio);
    mat4 rotation_matrix = get_y_rotation_matrix(0.0f);
    float mesh_position_offset[3] = {0.0f, -1.5f, -3.0f};
    float object_color[3] = {0.8f, 0.13f, 0.42f};
    float light_position[3] = {1.0f, 1.0f, 0.0f};
    glEnable(GL_DEPTH_TEST);
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        /*glBindVertexArray(gl_mesh.vao);
        glUseProgram(mesh_shader.shader_program);
        glUniformMatrix4fv(mesh_shader.perspective_matrix_location, 1, GL_FALSE, perspective_matrix.data);
        glUniformMatrix4fv(mesh_shader.rotation_matrix_location, 1, GL_FALSE, rotation_matrix.data);
        glUniform3fv(mesh_shader.position_offset_location, 1, mesh_position_offset);
        glUniform3fv(mesh_shader.object_color_location, 1, object_color);
        glUniform1f(mesh_shader.ambient_strength_location, 0.7f); 
        glUniform3fv(mesh_shader.light_position_location, 1, light_position);
        //glDrawElements(GL_TRIANGLES, mesh->num_vertices, GL_UNSIGNED_INT, NULL);
        //glDrawArrays(GL_TRIANGLES, 0, gl_mesh.num_vertices);
        */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
