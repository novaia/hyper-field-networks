import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import json
from fields import image_field

if __name__ == '__main__':
    weight_path = 'data/approximation_field/0_weights.npy'
    weight_map_path = 'data/approximation_field/weight_map.json'
    weight_map = json.load(open(weight_map_path, 'r'))
    weights = jnp.load(weight_path)
    
    model = image_field.ImageField(mlp_depth=3, mlp_width=64, encoding_dim=10)
    state = image_field.create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    
    print(weight_map)
    current_height = 0
    for i in range(len(weight_map)):
        layer_map = weight_map[i]
        layer_bias = weights[current_height, :layer_map['bias_width']]
        current_height += 1
        layer_kernel = weights[
            current_height:current_height+layer_map['kernel_height'], 
            :layer_map['kernel_width']
        ]
        if layer_map['kernel_transposed']:
            layer_kernel = jnp.transpose(layer_kernel)
        current_height += layer_map['kernel_height']

        layer_name = layer_map['layer']
        state.params[layer_name]['bias'] = layer_bias
        state.params[layer_name]['kernel'] = layer_kernel

        print(layer_name)
        print('Actual bias', state.params[layer_name]['bias'].shape)
        print('Loaded bias', layer_bias.shape)
        print('Actual kernel', state.params[layer_name]['kernel'].shape)
        print('Loaded kernel', layer_kernel.shape)
        print('\n')

    print('Parsed height', current_height)
    image_field.draw_image(state, 64, 64, 'data/generated_weights/test_image.png')
