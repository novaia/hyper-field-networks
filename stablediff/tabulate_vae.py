import jax
import jax.numpy as jnp
from .models.vae import AutoencoderKl
import json

def main():
    config_path = 'configs/stable_diffusion_2_vae.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = AutoencoderKl(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        down_block_types=config['down_block_types'],
        up_block_types=config['up_block_types'],
        layers_per_block=config['layers_per_block'],
        act_fn=config['act_fn'],
        latent_channels=config['latent_channels'],
        norm_num_groups=config['norm_num_groups'],
        sample_size=config['sample_size'],
        block_out_channels=config['block_out_channels']
    )

    sample_shape = (1, config['in_channels'], config['sample_size'], config['sample_size'])
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(key)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    print(model.tabulate(rngs, sample))

if __name__ == '__main__':
    main()