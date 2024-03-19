#version 330 core

in vec3 frag_normal;
in vec3 frag_pos;

void main()
{
    vec3 light_pos = normalize(vec3(1.0f, 0.2f, 1.0f));
    vec3 light_color = vec3(1.0f);
    vec3 object_color = vec3(0.99f, 0.62f, 0.33f);
    float diffuse_blend = 0.6f;
    float specular_blend = 1.0f - diffuse_blend;

    // Ambient lighting.
    float ambient_strength = 0.8f;
    vec3 ambient = ambient_strength * light_color;
        
    // Diffuse lighting.
    vec3 normal = normalize(frag_normal);
    vec3 light_dir = normalize(light_pos - normalize(frag_pos));
    float diffuse_strength = max(0.0f, dot(normal, light_dir));
    vec3 diffuse = light_color * diffuse_strength;

    // Specular lighting.
    vec3 view_dir = normalize(frag_pos);
    vec3 reflect_dir = normalize(reflect(light_dir, normal));
    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 64.0f);
    vec3 specular = specular_strength * light_color;

    vec3 color = (
        ambient 
        + (diffuse * diffuse_blend) 
        + (specular * specular_blend)
    ) * object_color;
    gl_FragColor = vec4(color, 1.0f);
}
