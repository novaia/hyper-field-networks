import numpy as jnp
import argparse
import os
import json

def _generate_leaf_map(leaf):
    leaf_map = 
    return leaf_map

def generate_weight_map(module):
    weight_map = {}
    for key in module.keys():
        sub_module = module[key]
        if isinstance(sub_module, dict):
            weight_map[key] = generate_weight_map(sub_module)
        else:
            weight_map[key] = {
                'shape': leaf.shape, 
                'flat_dim': jnp.ravel(leaf).shape[0]
            }
    return weight_map

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
