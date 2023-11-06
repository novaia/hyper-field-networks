from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
import random
from functools import partial
from hypernets.common.nn import LinearAttention, SinusoidalEmbedding
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion

class FeedForward(nn.Module):
    feed_forward_dim:int
    output_dim:int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.feed_forward_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.gelu(x)
        return x

class LinearTransformer(nn.Module):
    attention_dim:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int

    @nn.compact
    def __call__(self, x):
        RematDense = nn.remat(nn.Dense)
        RematLinearAttention = nn.remat(LinearAttention)
        RematFeedForward = nn.remat(FeedForward)

        x, noise_variances = x
        e = SinusoidalEmbedding(self.embedding_max_frequency)(noise_variances)
        x = jnp.concatenate([x, e], axis=-2)
        x = RematDense(features=self.embedding_dim)(x)
        positions = jnp.arange(self.context_length+1)
        e = nn.Embed(
            num_embeddings=self.context_length+1, features=self.embedding_dim
        )(positions)
        x = x + e

        for _ in range(self.num_bocks):
            residual = x
            x = RematLinearAttention(self.attention_dim, self.embedding_dim)(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = RematFeedForward(self.feed_forward_dim, self.embedding_dim)(x)
            x = nn.LayerNorm()(x + residual)

        x = nn.Dense(features=self.token_dim)(x)
        x = x[:, :-1, :] # Remove embedded noise variances token.
        return x

def train_step(state:TrainState, key:int, batch:jax.Array):
    noise_key, diffusion_time_key = jax.random.split(key)

    def loss_fn(params):
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(diffusion_times, 0.02, 0.95)
        noise = jax.random.normal(noise_key, batch.shape)
        noisy_batch = batch * signal_rates + noise * noise_rates
        x = (noisy_batch, noise_rates**2)
        pred_noise = state.apply_fn({'params': params}, x)
        return jnp.mean((pred_noise - noise)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def save_image(image, file_path):
    image = image - jnp.min(image)
    image = image / jnp.max(image)
    plt.imsave(file_path, image)

def load_dataset(path:str):
    dataset_directory_list = os.listdir(path)
    dataset = []
    for file in dataset_directory_list:
        if file.endswith('.npy'):
            dataset.append(jnp.load(os.path.join(path, file)))
    dataset = jnp.array(dataset, dtype=jnp.float32)
    return dataset

def load_batch(sample_paths, batch_size):
    random.shuffle(sample_paths)
    batch_paths = sample_paths[:batch_size]
    batch = []
    for path in batch_paths:
        batch.append(jnp.load(path))
    batch = jnp.array(batch, dtype=jnp.float32)
    return batch

def main():
    batch_size = 1
    datast_path = 'data/synthetic_nerfs/packed_aliens'
    all_sample_paths = os.listdir(datast_path)
    valid_sample_paths = []
    for path in all_sample_paths:
        if path.endswith('.npy'):
            full_path = os.path.join(datast_path, path)
            valid_sample_paths.append(full_path)
    del all_sample_paths
    load_batch_fn = partial(load_batch, sample_paths=valid_sample_paths, batch_size=batch_size)

    dummy_batch = load_batch_fn()
    print('batch shape', dummy_batch.shape)
    token_dim = dummy_batch.shape[-1]
    #context_length = dummy_batch.shape[-2]
    context_length = 120_000

    model = LinearTransformer(
        attention_dim=256,
        token_dim=token_dim,
        embedding_dim=256,
        num_bocks=2,
        feed_forward_dim=256,
        embedding_max_frequency=1000.0,
        context_length=context_length
    )

    tx = optax.adam(1e-3)
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones((1, context_length, token_dim)), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_steps = 100
    for step in range(train_steps):
        step_key = jax.random.PRNGKey(step)
        batch = load_batch_fn()
        batch, extra = jnp.split(batch, axis=1, indices_or_sections=[context_length])
        extra.delete()
        loss, state = train_step(state, step_key, batch)
        batch.delete()
        print('loss:', loss)
    print('Finished training')

    generation = reverse_diffusion(
        state.apply_fn, 
        state.params, 
        num_images=1, 
        diffusion_steps=20, 
        context_length=context_length,
        token_dim=token_dim,
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=0.02,
        max_signal_rate=0.95,
        seed=2
    )
    save_image(generation[0], 'data/generation.png')

if __name__ == '__main__':
    main()