#version 330 core

uniform float ambient_strength;
uniform sampler2D texture_sampler;

in vec3 frag_normal;
in vec3 frag_m_pos;
in vec3 frag_mv_pos;
in vec2 frag_texture_coord;
in vec3 frag_light_pos;

void main()
{
    vec3 light_pos = frag_light_pos;
    vec3 light_color = vec3(1.0f);
    float diffuse_blend = 1.0f;
    float specular_blend = 1.0f - diffuse_blend;

    // Ambient lighting.
    vec3 ambient = ambient_strength * light_color;
        
    // Diffuse lighting.
    vec3 normal = normalize(frag_normal);
    vec3 light_dir = normalize(light_pos - frag_m_pos);
    float diffuse_strength = max(0.0f, dot(normal, light_dir));
    vec3 diffuse = light_color * diffuse_strength;

    // Specular lighting.
    vec3 view_dir = normalize(frag_mv_pos);
    vec3 reflect_dir = normalize(reflect(light_dir, normal));
    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 64.0f);
    vec3 specular = specular_strength * light_color;
    
    vec4 texture_color = texture(texture_sampler, frag_texture_coord);
    vec3 color = (
        ambient 
        + (diffuse * diffuse_blend) 
        + (specular * specular_blend)
    ) * texture_color.rgb;
    gl_FragColor = vec4(color, 1.0f);
    gl_FragColor = vec4(normal, 1.0f);
    //gl_FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
