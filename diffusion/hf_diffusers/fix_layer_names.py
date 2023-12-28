# Temp script for fixing layer names.

from diffusion.hf_diffusers.models.unet_conditional import get_model_from_config

import jax
import jax.numpy as jnp

import json

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
    layer_name = 'down_blocks_0'
    layer_name2 = 'attentions_0'
    layer_name3 = 'transformer_blocks_0'
    layer_name4 = 'ff'
    print('Random params:\n', sorted(list(random_params[layer_name][layer_name2][layer_name3][layer_name4].keys())))
    #print('Random params:\n', sorted(list(random_params[layer_name][layer_name2][layer_name3].keys())))
    #print('Random params:\n', sorted(list(random_params[layer_name][layer_name2].keys())))
    #print('Random params:\n', sorted(list(random_params[layer_name].keys())))

    loaded_params = dict(jnp.load(model_path, allow_pickle=True).tolist())
    print('Loaded params:\n', sorted(list(loaded_params[layer_name][layer_name2][layer_name3][layer_name4].keys())))
    #print('Loaded params:\n', sorted(list(loaded_params[layer_name][layer_name2][layer_name3].keys())))
    #print('Loaded params:\n', sorted(list(loaded_params[layer_name][layer_name2].keys())))
    #print('Loaded params:\n', sorted(list(loaded_params[layer_name].keys())))

if __name__ == '__main__':
    main()