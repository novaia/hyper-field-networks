import jax
import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel

def main():
    in_channels = 4
    sample_size = 32
    cross_attention_dim = 1280
    sample_shape = (1, in_channels, sample_size, sample_size)
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    timesteps = jnp.ones((1,), dtype=jnp.int32)
    encoder_hidden_states = jnp.zeros((1, 1, cross_attention_dim), dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(key)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    model = UNet2dConditionalModel()
    print(model.tabulate(rngs, sample, timesteps, encoder_hidden_states))

if __name__ == '__main__':
    main()