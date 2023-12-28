# Temp script for fixing layer names.

from stablediff.models.unet_conditional import get_model_from_config

import jax
import jax.numpy as jnp

import json

def get_all_keys(keys, weight_dict, output_file):
    for key in keys:
        new_weight_dict = weight_dict[key]
        if isinstance(new_weight_dict, dict):
            get_all_keys(new_weight_dict.keys(), new_weight_dict, output_file)
        with open(output_file, 'a') as f:
            f.write(key + '\n')
    return

def main():
    config_path = 'configs/stable_diffusion_2_unet.json'
    model_path = 'data/models/stable_diffusion_2_unet_flax.npy'
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))
    
    dtype = jnp.float16
    model = get_model_from_config(config, dtype)
    key = jax.random.PRNGKey(0)
    sample_key, params_rng, dropout_rng = jax.random.split(key, 3)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    samples = jax.random.normal(
        sample_key, 
        (1, config['in_channels'], config['sample_size'], config['sample_size']),
        dtype=dtype
    )
    timesteps = jnp.ones((1,), dtype=dtype)
    encoder_hidden_states = jnp.zeros((1, 1, config['cross_attention_dim']), dtype=dtype)
    random_params = model.init(rngs, samples, timesteps, encoder_hidden_states)['params']
    loaded_params = dict(jnp.load(model_path, allow_pickle=True).tolist())
    get_all_keys(sorted(list(random_params.keys())), random_params, 'data/random_params_keys.txt')
    get_all_keys(sorted(list(loaded_params.keys())), loaded_params, 'data/loaded_params_keys.txt')

if __name__ == '__main__':
    main()