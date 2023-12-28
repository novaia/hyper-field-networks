import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel
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
    
    dtype = jnp.float16
    model = UNet2dConditionalModel(
        sample_size=config['sample_size'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        down_block_types=config['down_block_types'],
        up_block_types=config['up_block_types'],
        only_cross_attention=config['dual_cross_attention'],
        block_out_channels=config['block_out_channels'],
        layers_per_block=config['layers_per_block'],
        attention_head_dim=config['attention_head_dim'],
        cross_attention_dim=config['cross_attention_dim'],
        use_linear_projection=config['use_linear_projection'],
        flip_sin_to_cos=config['flip_sin_to_cos'],
        freq_shift=config['freq_shift'],
        dtype=dtype
    )
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
    print('Random params:\n', random_params.keys())

    loaded_params = dict(jnp.load(model_path, allow_pickle=True).tolist())
    print('Loaded params:\n', loaded_params.keys())
    
    #tx = optax.sgd(learning_rate=1.0)
    #state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    #y = state.apply_fn({'params': state.params}, sample, timesteps, encoder_hidden_states)
    #print('Output shape', y.shape)

if __name__ == '__main__':
    main()