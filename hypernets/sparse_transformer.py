import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt

def sinusoidal_embedding(x, embedding_max_frequency, embedding_dims):
    embedding_min_frequency = 1.0
    frequencies = jnp.exp(
        jnp.linspace(
            jnp.log(embedding_min_frequency),
            jnp.log(embedding_max_frequency),
            embedding_dims // 2
        )
    )
    angular_speeds = 2.0 * jnp.pi * frequencies
    embeddings = jnp.concatenate(
        [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
        axis = -1
    )
    return embeddings

class SparseTransformer(nn.Module):
    attention_dim:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int

    @nn.compact
    def __call__(self, x):
        @jax.custom_jvp
        def attention(x):
            query = nn.Dense(features=self.attention_dim)(x)
            key = nn.Dense(features=self.attention_dim)(x)
            value = nn.Dense(features=self.attention_dim)(x)
            x = nn.dot_product_attention(query, key, value)
            x = nn.Dense(features=self.embedding_dim)(x)
            x = nn.gelu(x)
            return x
        attention_fwd = lambda x: attention(x)
        attention_bwd = lambda x, g: jax.grad(attention)(x) * g
        attention.defjvp(attention_fwd, attention_bwd)

        x, noise_variances = x
        embeded_noise_variances = sinusoidal_embedding(
            noise_variances, self.embedding_max_frequency, self.token_dim
        )
        print(embeded_noise_variances.shape)
        x = jnp.concatenate([x, embeded_noise_variances], axis=-2)
        x = nn.Dense(features=self.embedding_dim)(x)
        positions = jnp.arange(self.context_length+1)
        embedded_position = nn.Embed(
            num_embeddings=self.context_length+1, features=self.embedding_dim
        )(positions)
        x = x + embedded_position

        for _ in range(self.num_bocks):
            residual = x
            x = attention(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.embedding_dim)(x)

        x = x[:, :-1, :] # Remove embedded noise variances token.
        x = nn.Dense(features=self.token_dim)(x)
        return x

def main():
    sequence_height = 300
    sequence_width = 40
    sequence = jnp.arange(sequence_height) * jnp.pi / 8
    sequence = jnp.sin(sequence)
    sequence = jnp.repeat(jnp.expand_dims(sequence, axis=-1), sequence_width, axis=-1)
    sequence = sequence - jnp.min(sequence)
    sequence = sequence / jnp.max(sequence)
    plt.imsave('data/sequence.png', sequence)
    sequence = jnp.expand_dims(sequence, axis=0)

    model = SparseTransformer(
        attention_dim=64,
        token_dim=sequence_width,
        embedding_dim=64,
        num_bocks=2,
        feed_forward_dim=256,
        embedding_max_frequency=10.0,
        context_length=sequence_height
    )
    tx = optax.adam(1e-3)
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones(sequence.shape), jnp.ones((sequence.shape[0], 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

if __name__ == '__main__':
    main()