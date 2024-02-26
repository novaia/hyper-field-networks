import numpy as jnp
import argparse
import os
import json

def generate_param_map(module, start_pos=0):
    param_map = {}
    for key in module.keys():
        sub_module = module[key]
        if isinstance(sub_module, dict):
            param_map[key], start_pos = generate_param_map(sub_module, start_pos)
        else:
            flat_dim = jnp.ravel(sub_module).shape[0]
            param_map[key] = {
                'shape': sub_module.shape, 
                'flat_dim': flat_dim,
                'start_pos': start_pos
            }
            start_pos += flat_dim
    # When recursion is finished, start_pos will be equal the number of params.
    return param_map, start_pos

def flatten_params(module, param_map, num_params):
    # Breaking recursion's purity by adding state here makes things simpler.
    flat_params = jnp.zeros((num_params))
    def __recurse(__module, __param_map):
        for key in __module.keys():
            sub_module = __module[key]
            sub_map = __param_map[key]
            if isinstance(sub_module, dict):
                __recurse(sub_module, sub_map)
            else:
                start_pos = sub_map['start_pos']
                flat_params[start_pos : start_pos + sub_map['flat_dim']] = jnp.ravel(sub_module)
    __recurse(module, param_map)
    return flat_params

def unflatten_params(flat_params, param_map):
    unflat_params = {}
    for key in param_map:
        sub_map = param_map[key]
        if 'start_pos' not in sub_map.keys():
            unflat_params[key] = unflatten_params(flat_params, sub_map)
        else:
            start_pos = sub_map['start_pos']
            unflat_params[key] = jnp.reshape(
                flat_params[start_pos : start_pos + sub_map['flat_dim']], 
                sub_map['shape']
            )
    return unflat_params

def main():
    test_tree = {
        'layer_a': {
            'layer_b': jnp.ones((5, 5))
        },
        'layer_c': jnp.ones((2, 3))
    }
    param_map, num_params = generate_param_map(test_tree) 
    print(json.dumps(param_map, indent=4))
    flat_params = flatten_params(test_tree, param_map, num_params)
    print(flat_params.shape)
    unflat_params = unflatten_params(flat_params, param_map)
    print(unflat_params)

if __name__ == '__main__':
    main()
