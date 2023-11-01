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
from typing import List, Tuple, Callable
from operator import itemgetter
import math

@dataclass
class DatasetInfo:
    file_paths: List[str]
    num_files: int
    num_tokens: int
    token_dim: int
    num_contexts: int
    batch_size: int
    context_length: int
    files_to_load: int # Number of files needed to fill a batch.
    samples_per_file: int # Number of samples to take from each file for batch.
    loaded_shape: Tuple[int, int, int] # Shape of loaded batch files.
    pad_shape: Tuple[int, int, int] # Shape of padding needed to split file into num_contexts.
    num_pad_tokens: int # Number of padding tokens needed to split file into num_contexts.
    final_shape: Tuple[int, int, int] # Shape of final batch.
    context_indices: jax.Array
    file_indices: jax.Array

def create_dataset_info(path:str, context_length:int, batch_size:int, verbose:bool):
    path_list = os.listdir(path)
    valid_file_paths = []
    for file_path in path_list:
        if file_path.endswith('.npy'):
            valid_file_paths.append(os.path.join(path, file_path))
    num_files = len(valid_file_paths)
    assert num_files > 0, 'No .npy files found in specified directory'
    base_sample = jnp.load(valid_file_paths[0])
    assert len(base_sample.shape) == 2, 'Samples arrays must be 2D'
    num_tokens = base_sample.shape[0]
    token_dim = base_sample.shape[1]
    num_pad_tokens = (context_length - (num_tokens % context_length)) % context_length
    num_contexts = (num_tokens + num_pad_tokens) // context_length
    context_indices = jnp.arange(context_length)
    file_indices = jnp.arange(num_files)
    files_to_load = math.ceil(batch_size / num_contexts)
    loaded_shape = (files_to_load, num_tokens, token_dim)
    pad_shape = (files_to_load, num_pad_tokens, token_dim)
    samples_per_file = batch_size // files_to_load
    final_shape = (batch_size, context_length, token_dim)

    if verbose:
        print('Number of files:', num_files)
        print('Number of tokens per sample:', num_tokens)
        print('Token dim:', token_dim)
        print('Context length:', context_length)
        print('Number of contexts per sample:', num_contexts)
        print('Loaded shape:', loaded_shape)
        print('Pad shape:', pad_shape)
        print('Files to load:', files_to_load)
        print('Batch size:', batch_size)
        print('Context length:', context_length)

    dataset_info = DatasetInfo(
        file_paths=valid_file_paths,
        num_files=num_files,
        num_tokens=num_tokens,
        token_dim=token_dim,
        loaded_shape=loaded_shape,
        pad_shape=pad_shape,
        final_shape=final_shape,
        num_contexts=num_contexts,
        files_to_load=files_to_load,
        samples_per_file=samples_per_file,
        batch_size=batch_size,
        context_length=context_length,
        context_indices=context_indices,
        file_indices=file_indices,
        num_pad_tokens=num_pad_tokens
    )
    return dataset_info

def load_batch(dataset_info:DatasetInfo, key):
    sample_key, context_key = jax.random.split(key, num=2)

    # Load all files required for the batch.
    file_indices = jax.random.choice(
        sample_key, 
        dataset_info.file_indices, 
        shape=(dataset_info.files_to_load,), 
        replace=False
    )
    batch_paths = itemgetter(*file_indices.tolist())(dataset_info.file_paths)
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
    batch = jnp.concatenate([jnp.zeros(dataset_info.pad_shape), batch], axis=1)

    # Split the batch into multiple contexts.
    split_contexts_shape = (
        dataset_info.files_to_load,
        dataset_info.num_contexts, 
        dataset_info.context_length, 
        dataset_info.token_dim
    )
    batch = jnp.reshape(batch, split_contexts_shape)

    # Select random contexts.
    context_indices = jax.random.choice(
        context_key,
        dataset_info.context_indices,
        shape=(dataset_info.samples_per_file,),
        replace=True
    )
    batch = batch[:, context_indices, :, :]
    batch = jnp.reshape(batch, dataset_info.final_shape)
    return batch, context_indices

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
        x, context_indices, noise_variances = x
        embedded_noise_variances = sinusoidal_embedding(
            noise_variances, self.embedding_max_frequency, self.token_dim
        )
        embedded_context_indices = sinusoidal_embedding(
            context_indices, self.embedding_max_frequency, self.token_dim
        )
        x = jnp.concatenate([x, embedded_noise_variances, embedded_context_indices], axis=-2)
        x = nn.Dense(features=self.embedded_token_dim)(x)
        positions = jnp.arange(self.context_length+2)
        embedded_position = nn.Embed(
            num_embeddings=self.context_length+2, features=self.embedded_token_dim
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
        x = x[:, :-2, :] # Remove embedded noise variances and context indices tokens.
        return x
    
def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def create_train_state(model, rng, learning_rate, context_length, token_dim, steps_per_epoch):
    x = (jnp.ones([1, context_length, token_dim]), jnp.ones([1, 1, 1]), jnp.ones([1, 1, 1]))
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def train_loop(
    datset_info:DatasetInfo, epochs:int, min_signal_rate:float, max_signal_rate:float, 
    state:TrainState
):
    steps_per_epoch = datset_info.num_files // datset_info.batch_size
    losses = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            losses_this_epoch = []
            step_key = jax.random.PRNGKey(epoch * steps_per_epoch + step)
            batch_key, train_key = jax.random.split(step_key, 2)
            batch, context_indices = load_batch(datset_info, batch_key)
            loss, state = train_step(
                batch=batch,
                context_indices=context_indices,
                min_signal_rate=min_signal_rate,
                max_signal_rate=max_signal_rate,
                state=state,
                parent_key=train_key
            )
            print('Step', step, 'Loss:', loss)
            losses_this_epoch.append(loss)
        losses.append(sum(losses_this_epoch) / len(losses_this_epoch))
        print('Epoch', epoch, 'Loss:', losses[-1])
    return state

@jax.jit
def train_step(
    batch:jax.Array, context_indices:jax.Array, min_signal_rate:float, max_signal_rate:float, 
    state:TrainState, parent_key
):
    noise_key, diffusion_time_key = jax.random.split(parent_key, 2)
    context_indices = jnp.reshape(context_indices, (context_indices.shape[0], 1, 1))

    def loss_fn(params):
        noises = jax.random.normal(noise_key, batch.shape)
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        noisy_batch = signal_rates * batch + noise_rates * noises
        x = [noisy_batch, context_indices, noise_rates**2]
        pred_noises = state.apply_fn({'params': params}, x)

        loss = jnp.mean((pred_noises - noises)**2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
    state = state.apply_gradients(grads=grads)
    return loss, state

def reverse_diffusion(
    dataset_info:DatasetInfo,
    apply_fn:Callable, 
    params,
    batch_size, 
    diffusion_steps, 
    context_length,
    token_dim, 
    diffusion_schedule_fn,
    min_signal_rate,
    max_signal_rate,
    seed, 
    initial_noise = None,
):
    assert batch_size <= dataset_info.num_contexts, 'Batch size must be <= num contexts'
    num_batches = math.ceil(dataset_info.num_contexts / batch_size)
    denoised_contexts = []
    for i in range(num_batches):
        if initial_noise == None:
            initial_noise = jax.random.normal(
                jax.random.PRNGKey(seed), 
                shape=(batch_size, context_length, token_dim)
            )
        step_size = 1.0 / diffusion_steps
        context_indices = jnp.arange(i * batch_size, (i+1) * batch_size)
        context_indices = jnp.reshape(context_indices, (batch_size, 1, 1))

        next_noisy_weights = initial_noise
        for step in range(diffusion_steps):
            noisy_weights = next_noisy_weights
            
            diffusion_times = jnp.ones((batch_size, 1, 1)) - step * step_size
            noise_rates, signal_rates = diffusion_schedule_fn(
                diffusion_times, min_signal_rate, max_signal_rate
            )
            x = [noisy_weights, context_indices, noise_rates**2]
            pred_noises = lax.stop_gradient(apply_fn({'params': params}, x))
            pred_weights = (noisy_weights - noise_rates * pred_noises) / signal_rates
            
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = diffusion_schedule_fn(
                next_diffusion_times, min_signal_rate, max_signal_rate
            )
            next_noisy_weights = \
                next_signal_rates * pred_weights + next_noise_rates * pred_noises
        print(f'Context {i * batch_size} to {(i+1) * batch_size} denoised')
        denoised_contexts.append(pred_weights)
    
    num_leftover_contexts = dataset_info.num_contexts % batch_size
    if num_leftover_contexts != 0:
        denoised_contexts[-1] = denoised_contexts[-1][:num_leftover_contexts, :, :]

    denoised_weights = jnp.concatenate(denoised_contexts, axis=0)
    new_shape = (
        denoised_weights.shape[0] * denoised_weights.shape[1],
        dataset_info.token_dim
    )
    denoised_weights = jnp.reshape(denoised_weights, new_shape)
    denoised_weights = denoised_weights[dataset_info.num_pad_tokens:, :]
    return denoised_weights

def main():
    print('GPU:', jax.devices('gpu'))

    epochs = 1
    batch_size = 8
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    embedding_max_frequency = 1000.0
    num_blocks = 4
    feed_forward_dim = 512
    attention_dim = 512
    embedded_token_dim = 512
    attention_heads = 8
    learning_rate = 5e-5
    context_length = 1024

    dataset_info = create_dataset_info(
        'data/synthetic_nerfs/packed_aliens', context_length, batch_size, verbose=True
    )
    batch, context_indices = load_batch(dataset_info, jax.random.PRNGKey(0))
    print('Batch shape', batch.shape)
    print('Batch context indices shape', context_indices.shape)
    token_dim = batch.shape[-1]
    steps_per_epoch = (dataset_info.num_files * dataset_info.num_tokens) // batch_size

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
    state = train_loop(
        dataset_info, 
        epochs,
        min_signal_rate, 
        max_signal_rate, 
        state
    )
    generated_weights = reverse_diffusion(
        dataset_info=dataset_info,
        apply_fn=state.apply_fn, 
        params=state.params, 
        batch_size=32, 
        diffusion_steps=20,
        context_length=context_length,
        token_dim=token_dim,
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        seed=0
    )
    print('Generated weights shape', generated_weights.shape)
    exit(0)
    generated_weights = jnp.squeeze(generated_weights)
    generated_weights = jnp.reshape(generated_weights, (generated_weights.shape[0], 48, 16))
    for i in range(generated_weights.shape[0]):
        jnp.save(f'data/generated_weights/{i}_weights.npy', generated_weights[i])
        weights_image = generated_weights[i] - jnp.min(generated_weights[i])
        weights_image = weights_image / jnp.max(weights_image)
        plt.imsave(f'data/generated_weights/{i}_weights.png', weights_image, cmap='magma')