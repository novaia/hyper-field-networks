import jax
import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel
import json

def main():
    config_path = 'configs/stable_diffusion_2_unet.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

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
        freq_shift=config['freq_shift']
    )

    sample_shape = (1, config['in_channels'], config['sample_size'], config['sample_size'])
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    timesteps = jnp.ones((1,), dtype=jnp.int32)
    encoder_hidden_states = jnp.zeros((1, 1, config['cross_attention_dim']), dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(key)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    print(model.tabulate(rngs, sample, timesteps, encoder_hidden_states))

if __name__ == '__main__':
    main()