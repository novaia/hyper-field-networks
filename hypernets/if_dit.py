# Image Field Diffusion Transformer.
# The purpose of this script is to test a full-attention diffusion transformer
# on the task of image field synthesis.
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from hypernets.common.nn import VanillaTransformer

from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types

from typing import Any, Callable
import json
import os

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_min_frequency:float = 1.0
    embedding_max_frequency:float = 1000.0
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
                dtype=self.dtype
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis = -1,
            dtype=self.dtype
        )
        return embeddings

class DiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    embedding_dim:int
    num_blocks:int
    feed_forward_dim:int
    token_dim:int
    context_length:int
    activation_fn:Callable
    normal_dtype:Any
    quantized_dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x, t):
        positions = jnp.arange(self.context_length+1)
        position_embedding = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        position_embedding = jnp.expand_dims(position_embedding, axis=0)
        
        time_embedding = SinusoidalEmbedding(
            self.embedding_dim,
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(t)
        #print('time_embedding', time_embedding.shape)
        time_embedding = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.Dense(
            self.embedding_dim,
            dtype=self.quantized_dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.LayerNorm()(time_embedding)
        
        skip = x
        skip_weight = nn.Dense(
            self.embedding_dim,
            dtype=self.quantized_dtype
        )(time_embedding)
        skip_weight = self.activation_fn(skip_weight)
        skip_weight = nn.Dense(1, dtype=self.quantized_dtype)(skip_weight)
        skip_weight = nn.sigmoid(skip_weight)

        #print('x', x.shape)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.LayerNorm()(x)
        #print('x', x.shape)

        # Add the diffusion time token to the end of the sequence.
        x = jnp.concatenate([x, time_embedding], axis=-2)
        x = x + position_embedding
        #print('x', x.shape)

        x = VanillaTransformer(
            num_blocks=self.num_blocks,
            attention_dim=self.attention_dim,
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim,
            feed_forward_dim=self.feed_forward_dim,
            activation_fn=self.activation_fn,
            normal_dtype=self.normal_dtype,
            quantized_dtype=self.quantized_dtype,
            remat=self.remat
        )(x)

        # Remove the diffusion time token from the end of the sequence.
        x = x[:, :-1, :]
        x = nn.Dense(
            features=self.token_dim, dtype=self.quantized_dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        x = x * (1 - skip_weight) + skip * (skip_weight)
        #print('output', x.shape)
        return x


def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def diffusion_schedule(t):
    start_angle = jnp.arccos(1.0)
    end_angle = jnp.arccos(0.0)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def ddim_sample(
    state, 
    num_samples:int, 
    diffusion_steps:int, 
    token_dim:int, 
    context_length:int,
    min_signal_rate:float,
    max_signal_rate:float,
    seed:int 
):
    @jax.jit
    def inference_fn(state, noisy_batch, diffusion_times):
        return jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_batch, diffusion_times)
        )
    
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_samples, context_length, token_dim)
    )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_batch = initial_noise
    for step in range(diffusion_steps):
        noisy_batch = next_noisy_batch
        
        diffusion_times = jnp.ones((num_samples, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = inference_fn(state, noisy_batch, noise_rates**2)
        pred_batch = (noisy_batch - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_batch = (next_signal_rates * pred_batch + next_noise_rates * pred_noises)
    return pred_batch

@jax.jit
def train_step(state, batch, batch_size, seed):
    noise_key, diffusion_time_key = jax.random.split(key, 2)
    noises = jax.random.normal(noise_key, batch.shape, dtype=jnp.float32)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch_size, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times)
    noisy_batch = signal_rates * batch + noise_rates * noises

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_images, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def get_data_iterator(dataset_path, token_dim, batch_size, num_threads=4):
    dataset_list = os.listdir(dataset_path)
    valid_dataset_list = [path for path in dataset_list if path.endswith('.npy')]
    assert len(valid_dataset_list) > 0, f'Could not find any .npy files in {dataset_path}'
    
    dummy_sample = jnp.load(os.path.join(dataset_path, valid_dataset_list[0]))
    print('Sample shape:', dummy_sample.shape)
    context_length = dummy_sample.shape[0]

    @pipeline_def
    def my_pipeline_def(file_root, context_length, token_dim):
        numpy_data = fn.readers.numpy(
            device='cpu', 
            file_root=file_root, 
            file_filter='*.npy', 
            shuffle_after_epoch=True,
            name='r'
        )
        numpy_data = numpy_data.gpu()
        tokens = fn.reshape(
            numpy_data, 
            shape=[context_length, token_dim], 
            device='gpu'
        )
        return tokens
    
    data_pipeline = my_pipeline_def(
        file_root=dataset_path,
        batch_size=batch_size,
        context_length=context_length,
        token_dim=token_dim,
        num_threads=num_threads, 
        device_id=0
    )
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], output_map=['x'], last_batch_policy=LastBatchPolicy.DROP
    )
    num_batches = len(valid_dataset_list) // batch_size
    return data_iterator, num_batches, context_length

def main():
    config_path = 'configs/if_dit.json'
    dataset_path = 'data/mnist_ingp_flat'

    with open('configs/if_dit.json', 'r') as f:
        config = json.load(f)
    token_dim = config['token_dim']
    batch_size = config['batch_size']

    data_iterator, steps_per_epoch, context_length = \
        get_data_iterator(dataset_path, token_dim, batch_size)
    print('Token dim:', token_dim)
    print('Context length:', context_length)
    print('Steps per epoch:', steps_per_epoch)

if __name__ == '__main__':
    main()
