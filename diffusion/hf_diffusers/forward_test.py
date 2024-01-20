import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel, get_model_from_config, init_model_params
import json

import jax
from jax.random import PRNGKey

from flax.training.train_state import TrainState
import optax

def main():
    config_path = 'configs/stable_diffusion_2_unet.json'
    model_path = 'data/models/stable_diffusion_2_unet_flax.npy'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))
    
    key = jax.random.PRNGKey(0)
    sample_key, param_key = jax.random.split(key)

    dtype = jnp.float16
    model = get_model_from_config(config, dtype)
    
    random_params = init_model_params(model, config, param_key)
    print('Random params:\n', random_params.keys())
    
    loaded_params = dict(jnp.load(model_path, allow_pickle=True).tolist())
    print('Loaded params:\n', loaded_params.keys())
 
    samples = jax.random.normal(
        sample_key, 
        (1, config['in_channels'], config['sample_size'], config['sample_size']),
        dtype=dtype
    )
    timesteps = jnp.ones((1,), dtype=dtype)
    encoder_hidden_states = jnp.zeros((1, 1, config['cross_attention_dim']), dtype=dtype)  
    tx = optax.sgd(learning_rate=1.0)
    state = TrainState.create(apply_fn=model.apply, params=loaded_params, tx=tx)
    y = jax.jit(state.apply_fn)({'params': state.params}, samples, timesteps, encoder_hidden_states)
    print('Output shape', y.shape)

if __name__ == '__main__':
    main()
