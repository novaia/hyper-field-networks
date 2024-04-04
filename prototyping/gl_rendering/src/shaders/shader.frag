#version 330 core

uniform float ambient_strength;
uniform sampler2D texture_sampler;
uniform sampler2D depth_map_sampler;

in vec3 frag_normal;
in vec3 frag_m_pos;
in vec3 frag_mv_pos;
in vec2 frag_texture_coord;
in vec3 frag_light_direction;

in vec4 frag_light_space_pos;

void main()
{
    vec3 light_color = vec3(1.0f);
    float diffuse_blend = 0.5f;
    float specular_blend = 1.0f - diffuse_blend;

    // Ambient lighting.
    vec3 ambient = ambient_strength * light_color;
        
    // Diffuse lighting.
    vec3 normal = normalize(frag_normal);
    vec3 light_direction = normalize(frag_light_direction);
    float diffuse_strength = max(0.0f, dot(normal, light_direction));
    vec3 diffuse = light_color * diffuse_strength;

    // Specular lighting.
    vec3 specular = vec3(0.0f);
    if(diffuse_strength > 0.0f)
    {
        vec3 view_dir = normalize(frag_mv_pos);
        vec3 reflect_dir = normalize(reflect(light_direction, normal));
        float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0f), 64.0f);
        specular = specular_strength * light_color;
    }
    
    // Shadow test.
    vec3 light_space_pos = frag_light_space_pos.xyz / frag_light_space_pos.w;
    light_space_pos = (light_space_pos * 0.5f) + 0.5f;
    float current_depth = light_space_pos.z;
    float closest_depth = texture(depth_map_sampler, light_space_pos.xy).r;
    float bias = max(0.005 * (1.0 - dot(normal, -light_direction)), 0.0005);  
    //float shadow = current_depth - bias > closest_depth ? 0.0f : 1.0f;
    
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(depth_map_sampler, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcf_depth = texture(depth_map_sampler, light_space_pos.xy + vec2(x, y) * texel_size).r; 
            shadow += current_depth - bias > pcf_depth ? 0.0 : 1.0;        
        }    
    }
    shadow /= 9.0;
    shadow = current_depth > 1.0f ? 1.0f : shadow;
    
    vec4 texture_color = texture(texture_sampler, frag_texture_coord);
    vec3 color = (
        ambient 
        + shadow * ((diffuse * diffuse_blend) + (specular * specular_blend))
    ) * texture_color.rgb;
    //vec3 color = (ambient + shadow) * texture_color.rgb;
    //color = vec3(shadow);
    gl_FragColor = vec4(color, 1.0f);
}
