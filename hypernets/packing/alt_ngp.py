import numpy as np
from jax import numpy as jnp
import jax
from fields import ngp_image
import matplotlib.pyplot as plt
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
            flat_dim = np.ravel(sub_module).shape[0]
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
    flat_params = np.zeros((num_params))
    def __recurse(__module, __param_map):
        for key in __module.keys():
            sub_module = __module[key]
            sub_map = __param_map[key]
            if isinstance(sub_module, dict):
                __recurse(sub_module, sub_map)
            else:
                start_pos = sub_map['start_pos']
                flat_params[start_pos : start_pos + sub_map['flat_dim']] = np.ravel(sub_module)
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
            # Params must be cast back to jax array in order to be used by model.
            unflat_params[key] = jnp.array(np.reshape(
                flat_params[start_pos : start_pos + sub_map['flat_dim']], 
                sub_map['shape']
            ))
    return unflat_params

def test():
    with open('configs/ngp_image.json', 'r') as f:
        config = json.load(f)
    params = dict(jnp.load('data/mnist_ingp/0.npy', allow_pickle=True).tolist())
    param_map, num_params = generate_param_map(params)
    print(json.dumps(param_map, indent=4))
    flat_params = flatten_params(params, param_map, num_params)
    unflat_params = unflatten_params(flat_params, param_map)
    
    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, 1e-2, jax.random.PRNGKey(0))
    state = state.replace(params = unflat_params)
    rendered_image = ngp_image.render_image(state, 28, 28)
    rendered_image = jax.device_put(rendered_image, jax.devices('cpu')[0])
    plt.imsave('data/test_render.png', rendered_image)

if __name__ == '__main__':
    test()
