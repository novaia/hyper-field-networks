import numpy as jnp
import argparse
import os
import json

def generate_weight_map(module, start_pos=0):
    weight_map = {}
    for key in module.keys():
        sub_module = module[key]
        if isinstance(sub_module, dict):
            weight_map[key], start_pos = generate_weight_map(sub_module, start_pos)
        else:
            flat_dim = jnp.ravel(sub_module).shape[0]
            weight_map[key] = {
                'shape': sub_module.shape, 
                'flat_dim': flat_dim,
                'start_pos': start_pos
            }
            start_pos += flat_dim
    return weight_map, start_pos

def main():
    test_tree = {
        'layer_a': {
            'layer_b': jnp.ones((5, 5))
        },
        'layer_c': jnp.ones((2, 3))
    }
    print(json.dumps(generate_weight_map(test_tree), indent=4))

if __name__ == '__main__':
    main()
