# Sliding Window Linear Transform DDIM.
from typing import Any, List, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
import math
import random
from operator import itemgetter
from functools import partial
from dataclasses import dataclass
from hypernets.common.nn import SinusoidalEmbedding, LinearTransformer
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion

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
    context_indices = jnp.arange(num_contexts)
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

def load_batch(dataset_info:DatasetInfo, key, dtype):
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
    batch = jnp.array(batch, dtype=dtype)
    batch = jnp.reshape(batch, dataset_info.loaded_shape)

    # Pad the batch.
    batch = jnp.concatenate([jnp.zeros(dataset_info.pad_shape, dtype=dtype), batch], axis=1)
    # Split the batch into multiple contexts.
    batch = jnp.split(batch, dataset_info.num_contexts, axis=1)
    batch = jnp.stack(batch, axis=1)
    # Select contexts for the batch.
    batch_context_indices = jax.random.choice(
        context_key, 
        dataset_info.context_indices, 
        shape=(dataset_info.samples_per_file,)
    )
    batch = batch[:, batch_context_indices]
    batch = jnp.reshape(batch, dataset_info.final_shape)
    return batch, batch_context_indices

class SlidingLTD(nn.Module):
    attention_dim:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int
    normal_dtype:Any
    quantized_dtype:Any

    @nn.compact
    def __call__(self, x):
        x, noise_variances, context_indices = x
        e_noise = SinusoidalEmbedding(
            self.token_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(noise_variances)
        e_context = SinusoidalEmbedding(
            self.token_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(context_indices)
        x = jnp.concatenate([x, e_noise, e_context], axis=-2)
        x = nn.remat(nn.Dense)(features=self.embedding_dim, dtype=self.quantized_dtype)(x)
        positions = jnp.arange(self.context_length+2)
        e = nn.Embed(
            num_embeddings=self.context_length+2, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        x = x + e

        x = LinearTransformer(
            num_blocks=self.num_bocks, 
            attention_dim=self.attention_dim, 
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype
        )(x)

        x = nn.remat(nn.Dense)(features=self.token_dim, dtype=self.quantized_dtype)(x)
        x = x[:, :-2, :] # Remove embedded noise variances and context indices.
        return x

def train_step(state:TrainState, key:int, batch:jax.Array, context_indices:jax.Array):
    noise_key, diffusion_time_key = jax.random.split(key)
    context_indices = jnp.reshape(context_indices, (batch.shape[0], 1, 1))

    def loss_fn(params):
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(diffusion_times, 0.02, 0.95)
        noise = jax.random.normal(noise_key, batch.shape)
        noisy_batch = batch * signal_rates + noise * noise_rates
        x = (noisy_batch, noise_rates**2, context_indices)
        pred_noise = state.apply_fn({'params': params}, x)
        return jnp.mean((pred_noise - noise)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
    state = state.apply_gradients(grads=grad)
    return loss, state

def save_image(image, file_path):
    image = image - jnp.min(image)
    image = image / jnp.max(image)
    plt.imsave(file_path, image)

def main():
    normal_dtype = jnp.float32
    quantized_dtype = jnp.float16
    batch_size = 1
    datast_path = 'data/synthetic_nerfs/packed_aliens'
    context_length = 168_000

    dataset = create_dataset_info(datast_path, 168_000, batch_size, verbose=True)
    dummy_batch, context_index = load_batch(dataset, jax.random.PRNGKey(0), quantized_dtype)
    print('Dummy batch shape:', dummy_batch.shape)
    print('Dummy batch dtype:', dummy_batch.dtype)
    print('Context index:', context_index[0])
    token_dim = dummy_batch.shape[-1]
    dummy_batch.delete()
    context_index.delete()

    model = SlidingLTD(
        attention_dim=256,
        token_dim=token_dim,
        embedding_dim=256,
        num_bocks=2,
        feed_forward_dim=256,
        embedding_max_frequency=1000.0,
        context_length=context_length,
        normal_dtype=normal_dtype,
        quantized_dtype=quantized_dtype
    )

    tx = optax.adam(1e-3)
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones((1, context_length, token_dim)), jnp.ones((1, 1, 1)), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_steps = 100
    for step in range(train_steps):
        step_key = jax.random.PRNGKey(step)
        train_key, batch_key = jax.random.split(step_key)
        batch, context_indices = load_batch(dataset, batch_key, quantized_dtype)
        loss, state = train_step(state, train_key, batch, context_indices)
        batch.delete()
        print('loss:', loss)
    print('Finished training')

if __name__ == '__main__':
    main()