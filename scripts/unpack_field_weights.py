import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import json
from fields import image_field
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_weight_path = 'data/generated_weights/1_weights.npy'
    default_weight_map_path = 'data/approximation_field_small/weight_map.json'
    parser.add_argument('--weight_path', type=str, default=default_weight_path)
    parser.add_argument('--weight_map_path', type=str, default=default_weight_map_path)
    parser.add_argument('--mlp_width', type=int, default=64)
    parser.add_argument('--mlp_depth', type=int, default=3)
    parser.add_argument('--encoding_dim', type=int, default=10)
    args = parser.parse_args()

    weight_map = json.load(open(args.weight_map_path, 'r'))
    weights = jnp.load(args.weight_path)
    model = image_field.ImageField(
        mlp_depth=args.mlp_depth, mlp_width=args.mlp_width, encoding_dim=args.encoding_dim
    )
    state = image_field.create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    print('Packed weights shape:', weights.shape)
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
