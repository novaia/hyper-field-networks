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
    
    # TODO: some config settings are hardcoded and need to be made into parameters.
    model = UNet2dConditionalModel(
        sample_size=config['sample_size'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        down_block_types=config['down_block_types'],
        up_block_types=config['up_block_types'],
        only_cross_attention=config['dual_cross_attention'], #TODO: double check this.
        block_out_channels=config['block_out_channels'],
        layers_per_block=config['layers_per_block'],
        attention_head_dim=config['attention_head_dim'],
        cross_attention_dim=config['cross_attention_dim'],
        use_linear_projection=config['use_linear_projection'],
        flip_sin_to_cos=config['flip_sin_to_cos'],
        freq_shift=config['freq_shift'],
        dtype=jnp.float16
    )
    params = dict(jnp.load(model_path, allow_pickle=True).tolist())
    print(params.keys())
    tx = optax.sgd(learning_rate=1.0)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

if __name__ == '__main__':
    main()