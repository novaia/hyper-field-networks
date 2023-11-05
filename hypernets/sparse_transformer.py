import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

class SparseTransformer(nn.Module):
    attention_dim:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int
    num_heads:int=1

    @nn.compact
    def __call__(self, x):
        def sinusoidal_embedding(x):
            embedding_min_frequency = 1.0
            frequencies = jnp.exp(
                jnp.linspace(
                    jnp.log(embedding_min_frequency),
                    jnp.log(self.embedding_max_frequency),
                    self.token_dim // 2
                )
            )
            angular_speeds = 2.0 * jnp.pi * frequencies
            embeddings = jnp.concatenate(
                [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
                axis = -1
            )
            return embeddings

        x, noise_variances = x
        embeded_noise_variances = sinusoidal_embedding(noise_variances)
        x = jnp.concatenate([x, embeded_noise_variances], axis=-2)
        x = nn.Dense(features=self.embedding_dim)(x)
        positions = jnp.arange(self.context_length+1)
        embedded_position = nn.Embed(
            num_embeddings=self.context_length+1, features=self.embedding_dim
        )(positions)
        x = x + embedded_position

        RematAttention = nn.remat(nn.SelfAttention)
        for _ in range(self.num_bocks):
            residual = x
            x = RematAttention(
                num_heads=self.num_heads, 
                qkv_features=self.attention_dim,
                out_features=self.embedding_dim
            )(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.embedding_dim)(x)

        x = x[:, :-1, :] # Remove embedded noise variances token.
        x = nn.Dense(features=self.token_dim)(x)
        return x

def train_step(state:TrainState, key:int, batch:jax.Array):
    noise_key, diffusion_time_key = jax.random.split(key)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times, 0.01, 0.99)
    noise = jax.random.normal(noise_key, batch.shape)
    noisy_batch = batch * signal_rates + noise * noise_rates

    def loss_fn(params):
        x = (noisy_batch, noise_rates**2)
        y = state.apply_fn({'params': params}, x)
        return jnp.mean((y - batch)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def save_image(image, file_path):
    image = image - jnp.min(image)
    image = image / jnp.max(image)
    plt.imsave(file_path, image)

def main():
    sequence_height = 10000
    sequence_width = 64
    sequence = jnp.arange(sequence_height) * jnp.pi / 16
    sequence = jnp.sin(sequence)
    sequence = jnp.repeat(jnp.expand_dims(sequence, axis=-1), sequence_width, axis=-1)
    save_image(sequence, 'data/sequence.png')
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
    x = (jnp.ones(sequence.shape), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    batch_size = 1
    train_steps = 100
    for step in range(train_steps):
        step_key = jax.random.PRNGKey(step)
        batch = sequence
        loss, state = train_step(state, step_key, batch)
        print('loss:', loss)

if __name__ == '__main__':
    main()