import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from functools import partial
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from operator import itemgetter
import math


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
    
class HyperDiffusion(nn.Module):
    num_blocks: int
    feed_forward_dim: int
    attention_dim: int
    attention_heads: int
    token_dim: int
    embedded_token_dim: int
    embedding_max_frequency: float
    context_length: int

    @nn.compact
    def __call__(self, x):
        x, noise_variances = x
        embeded_noise_variances = sinusoidal_embedding(
            noise_variances, self.embedding_max_frequency, self.token_dim
        )
        x = jnp.concatenate([x, embeded_noise_variances], axis=-2)
        x = nn.Dense(features=self.embedded_token_dim)(x)
        positions = jnp.arange(self.context_length+1)
        embedded_position = nn.Embed(
            num_embeddings=self.context_length+1, features=self.embedded_token_dim
        )(positions)
        x = x + embedded_position
        for _ in range(self.num_blocks):
            residual = x
            x = nn.SelfAttention(
                num_heads=self.attention_heads, 
                qkv_features=self.attention_dim,
                out_features=self.embedded_token_dim
            )(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = nn.activation.relu(x)
            x = nn.Dense(features=self.embedded_token_dim)(x)
            x = nn.activation.relu(x)
            x = nn.LayerNorm()(x + residual)
        x = nn.Dense(features=self.token_dim)(x)
        x = x[:, :-1, :] # Remove embedded noise variances token.
        return x
    
def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def create_train_state(model, rng, learning_rate, context_length, token_dim, steps_per_epoch):
    x = (jnp.ones([1, context_length, token_dim]), jnp.ones([1, 1, 1]))
    variables = model.init(rng, x)
    params = variables['params']
    learning_rate_schedule = optax.exponential_decay(
        learning_rate, transition_begin=80*steps_per_epoch, 
        transition_steps=100*steps_per_epoch, decay_rate=0.8
    )
    tx = optax.adam(learning_rate_schedule)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state):
    steps_per_epoch = dataset.shape[0] // batch_size
    losses = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            losses_this_epoch = []
            step_key = jax.random.PRNGKey(epoch * steps_per_epoch + step)
            batch_indices_key, step_key = jax.random.split(step_key, num=2)
            batch_indices = jax.random.randint(
                batch_indices_key, shape=(batch_size,), minval=0, maxval=dataset.shape[0]
            )
            batch = dataset[batch_indices]
            loss, state = train_step(
                batch=batch, 
                min_signal_rate=min_signal_rate,
                max_signal_rate=max_signal_rate,
                state=state,
                parent_key=step_key
            )
            #print('Step', step, 'Loss:', loss)
            losses_this_epoch.append(loss)
        losses.append(sum(losses_this_epoch) / len(losses_this_epoch))
        print('Epoch', epoch, 'Loss:', losses[-1])
    return state

@jax.jit
def train_step(batch, min_signal_rate, max_signal_rate, state, parent_key):
    noise_key, diffusion_time_key = jax.random.split(parent_key, 2)
    
    def loss_fn(params):
        noises = jax.random.normal(noise_key, batch.shape)
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        noisy_batch = signal_rates * batch + noise_rates * noises

        pred_noises = state.apply_fn({'params': params}, [noisy_batch, noise_rates**2])

        loss = jnp.mean((pred_noises - noises)**2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
    state = state.apply_gradients(grads=grads)
    return loss, state

def reverse_diffusion(
    apply_fn, 
    params,
    num_images, 
    diffusion_steps, 
    context_length,
    token_dim, 
    diffusion_schedule_fn,
    min_signal_rate,
    max_signal_rate,
    seed, 
    initial_noise = None,
):
    if initial_noise == None:
        initial_noise = jax.random.normal(
            jax.random.PRNGKey(seed), 
            shape=(num_images, context_length, token_dim)
        )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule_fn(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = lax.stop_gradient(
            apply_fn(
                {'params': params}, 
                [noisy_images, noise_rates**2], 
            )
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule_fn(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

    return pred_images

def load_dataset(path:str, context_length:int):
    dataset_directory_list = os.listdir(path)
    dataset = []
    for file in dataset_directory_list:
        if file.endswith('.npy'):
            dataset.append(jnp.load(os.path.join(path, file)))
            break
    dataset = jnp.array(dataset, dtype=jnp.float32)
    pad_size = (context_length - (dataset.shape[1] % context_length)) % context_length
    dataset = jnp.concatenate([
        jnp.zeros((dataset.shape[0], pad_size, dataset.shape[2])),
        dataset,
    ], axis=1)
    num_samples = dataset.shape[0]
    num_contexts = dataset.shape[1] // context_length
    token_dim = dataset.shape[2]
    dataset = jnp.reshape(dataset, (num_samples, num_contexts, context_length, token_dim))
    return dataset

@dataclass
class DatasetInfo:
    sample_paths: List[str]
    num_samples: int
    num_tokens: int
    token_dim: int
    loaded_shape: Tuple[int, int]
    pad_shape: Tuple[int, int]
    num_contexts: int
    samples_to_load: int
    batch_size: int
    context_length: int
    context_indices: jax.Array
    sample_indices: jax.Array

def create_dataset_info(path:str, context_length:int, batch_size:int, verbose:bool):
    path_list = os.listdir(path)
    sample_paths = []
    for file_path in path_list:
        if file_path.endswith('.npy'):
            sample_paths.append(os.path.join(path, file_path))
    num_samples = len(sample_paths)
    assert num_samples > 0, 'No .npy files found in specified directory'
    base_sample = jnp.load(sample_paths[0])
    assert len(base_sample.shape) == 2, 'Samples arrays must be 2D'
    num_tokens = base_sample.shape[0]
    token_dim = base_sample.shape[1]
    pad_size = (context_length - (num_tokens % context_length)) % context_length
    pad_shape = (pad_size, token_dim)
    num_contexts = (num_tokens + pad_size) // context_length
    context_indices = jnp.arange(context_length)
    sample_indices = jnp.arange(num_samples)
    samples_to_load = math.ceil(batch_size / num_contexts)
    loaded_shape = (samples_to_load * num_tokens, token_dim)

    if verbose:
        print('Number of samples:', num_samples)
        print('Number of tokens per sample:', num_tokens)
        print('Token dim:', token_dim)
        print('Context length:', context_length)
        print('Number of contexts per sample:', num_contexts)
        print('Loaded shape:', loaded_shape)
        print('Pad shape:', pad_shape)
        print('Samples to load:', samples_to_load)
        print('Batch size:', batch_size)
        print('Context length:', context_length)

    dataset_info = DatasetInfo(
        sample_paths=sample_paths,
        num_samples=num_samples,
        num_tokens=num_tokens,
        token_dim=token_dim,
        loaded_shape=loaded_shape,
        pad_shape=pad_shape,
        num_contexts=num_contexts,
        samples_to_load=samples_to_load,
        batch_size=batch_size,
        context_length=context_length,
        context_indices=context_indices,
        sample_indices=sample_indices,
    )
    return dataset_info

def load_batch(dataset_info:DatasetInfo, key):
    sample_key, context_key = jax.random.split(key, num=2)

    # Load all files required for the batch.
    sample_indices = jax.random.choice(
        sample_key, 
        dataset_info.num_samples, 
        shape=(dataset_info.samples_to_load,), 
        replace=False
    )
    batch_paths = itemgetter(*sample_indices.tolist())(dataset_info.sample_paths)
    if type(batch_paths) == str:
        batch_paths = [batch_paths]
    else:
        batch_paths = list(batch_paths)
    batch = []
    for path in batch_paths:
        batch.append(jnp.load(path))
    batch = jnp.array(batch, dtype=jnp.float32)
    batch = jnp.reshape(batch, dataset_info.loaded_shape)

    # Pad the batch.
    batch = jnp.concatenate([jnp.zeros(dataset_info.pad_shape), batch], axis=0)

    # Split the batch into multiple contexts.
    split_contexts_shape = (
        dataset_info.num_contexts, 
        dataset_info.context_length, 
        dataset_info.token_dim
    )
    batch = jnp.reshape(batch, split_contexts_shape)

    # Select random contexts.
    context_indices = jax.random.choice(
        context_key,
        dataset_info.context_indices,
        shape=(dataset_info.batch_size,),
        replace=True
    )
    batch = batch[:, context_indices, :]
    return batch

def main():
    print('GPU:', jax.devices('gpu'))

    epochs = 1000
    batch_size = 32
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    embedding_max_frequency = 1000.0
    num_blocks = 1
    feed_forward_dim = 512
    attention_dim = 512
    embedded_token_dim = 512
    attention_heads = 8
    learning_rate = 5e-5
    context_length = 2048

    datast_info = create_dataset_info(
        'data/synthetic_nerfs/packed_aliens', context_length, batch_size, verbose=True
    )
    batch = load_batch(datast_info, jax.random.PRNGKey(0))
    print(batch.shape)
    exit(0)

    model = HyperDiffusion(
        num_blocks=num_blocks,
        feed_forward_dim=feed_forward_dim,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        token_dim=token_dim,
        embedded_token_dim=embedded_token_dim,
        embedding_max_frequency=embedding_max_frequency,
        context_length=context_length
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        model, rng, learning_rate, context_length, token_dim, steps_per_epoch
    )
    del rng
    state = train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state)
    generated_weights = reverse_diffusion(
        apply_fn=state.apply_fn, 
        params=state.params, 
        num_images=4, 
        diffusion_steps=20,
        context_length=context_length,
        token_dim=token_dim,
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        seed=0
    )
    exit(0)
    generated_weights = jnp.squeeze(generated_weights)
    generated_weights = jnp.reshape(generated_weights, (generated_weights.shape[0], 48, 16))
    for i in range(generated_weights.shape[0]):
        jnp.save(f'data/generated_weights/{i}_weights.npy', generated_weights[i])
        weights_image = generated_weights[i] - jnp.min(generated_weights[i])
        weights_image = weights_image / jnp.max(weights_image)
        plt.imsave(f'data/generated_weights/{i}_weights.png', weights_image, cmap='magma')