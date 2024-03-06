# Image Field Diffusion Transformer.
# The purpose of this script is the same as if_dit.py but with a customized
# rematerialization scheme in order to allow training larger models.
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from hypernets.common.nn import VanillaTransformer
from hypernets.packing.alt_ngp import unflatten_params
from fields import ngp_image
from matplotlib import pyplot as plt
import wandb

from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types

from functools import partial
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
    dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x, t):
        num_tokens = self.context_length + 1
        positions = jnp.reshape(jnp.arange(num_tokens), (1, num_tokens, 1))
        position_embedding = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            dtype=self.dtype
        )(positions)
        #print('position_embedding', position_embedding.shape)

        time_embedding = SinusoidalEmbedding(
            self.embedding_dim,
            dtype=self.dtype
        )(t)

        #print('x', x.shape)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.dtype
        )(x)
        x = self.activation_fn(x)
        #print('x', x.shape)

        # Add the diffusion time token to the end of the sequence.
        time_embedding = jnp.expand_dims(time_embedding, axis=1)
        x = jnp.concatenate([x, time_embedding], axis=-2)
        x = x + position_embedding
        #print('x', x.shape)
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.RMSNorm()(x)
            x = nn.remat(nn.MultiHeadDotProductAttention)(
                num_heads=self.num_attention_heads,
                dtype=self.dtype,
                qkv_features=self.attention_dim,
                out_features=self.embedding_dim
            )(inputs_q=x, inputs_kv=x)
            x = x + residual
            residual = x
            x = nn.RMSNorm()(x)
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = self.activation_fn(x)
            x = nn.Dense(features=self.embedding_dim)(x)
            x = x + residual

        # Remove the diffusion time token from the end of the sequence.
        x = x[:, :-1, :]
        x = nn.Dense(
            features=self.token_dim, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        #print('output', x.shape)
        return x


def diffusion_schedule(t):
    start_angle = jnp.arccos(0.999)
    end_angle = jnp.arccos(0.001)

    diffusion_angles = start_angle + t * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def ddim_sample(
    state, 
    num_samples:int, 
    diffusion_steps:int, 
    token_dim:int, 
    context_length:int,
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
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        pred_noises = inference_fn(state, noisy_batch, noise_rates**2)
        pred_batch = (
            (noisy_batch - jnp.expand_dims(noise_rates, axis=1) * pred_noises) 
            / jnp.expand_dims(signal_rates, axis=1)
        )
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(next_diffusion_times)
        next_noisy_batch = (
            jnp.expand_dims(next_signal_rates, axis=1) * pred_batch 
            + jnp.expand_dims(next_noise_rates, axis=1) * pred_noises
        )
    return pred_batch

@partial(jax.jit, static_argnames=['batch_size'])
def train_step(state, batch, batch_size, seed):
    noise_key, diffusion_time_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    noises = jax.random.normal(noise_key, batch.shape, dtype=jnp.float32)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch_size, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times)
    noisy_batch = (
        jnp.expand_dims(signal_rates, axis=1) * batch 
        + jnp.expand_dims(noise_rates, axis=1) * noises
    )

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_batch, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def train_loop(
    state, num_epochs, steps_per_epoch, data_iterator, 
    batch_size, context_length, token_dim, output_dir,
    field_state, image_width, image_height, field_param_map
):
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for step in range(steps_per_epoch):
            batch = next(data_iterator)['x']
            loss, state = train_step(state, batch, batch_size, state.step)
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        #print(average_loss)
        wandb.log({'loss': average_loss}, state.step)
        samples = ddim_sample(
            state=state,
            num_samples=10,
            diffusion_steps=20,
            token_dim=token_dim,
            context_length=context_length,
            seed=0
        )
        for i, sample in enumerate(samples):
            field_params = unflatten_params(flat_params=sample, param_map=field_param_map)
            field_state = field_state.replace(params=field_params)
            field_render = ngp_image.render_image(field_state, image_height, image_width)
            field_render = jax.device_put(field_render, jax.devices('cpu')[0])
            plt.imsave(os.path.join(output_dir, f'epoch{epoch}_image{i}.png'), field_render)

def get_data_iterator(dataset_path, token_dim, batch_size, num_threads=4):
    dataset_list = os.listdir(dataset_path)
    valid_dataset_list = [path for path in dataset_list if path.endswith('.npy')]
    assert len(valid_dataset_list) > 0, f'Could not find any .npy files in {dataset_path}'
    
    dummy_sample = jnp.load(os.path.join(dataset_path, valid_dataset_list[0]))
    print('Sample shape:', dummy_sample.shape)
    context_length = int(dummy_sample.shape[0] / token_dim)

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
    output_dir = 'data/if_dit_runs/6'
    config_path = 'configs/if_dit_remat.json'
    field_config_path = 'configs/ngp_image.json'
    dataset_path = 'data/mnist_ingp_flat'
    num_epochs = 1000
    # These are the dimensions of the images encoded by the neural fields.
    # The neural field dataset is trained on MNIST so the dimensions are 28x28.
    image_width = 28
    image_height = 28

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(field_config_path, 'r') as f:
        field_config = json.load(f)
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        field_param_map = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)
    token_dim = config['token_dim']
    batch_size = config['batch_size']

    data_iterator, steps_per_epoch, context_length = \
        get_data_iterator(dataset_path, token_dim, batch_size)
    print('Token dim:', token_dim)
    print('Context length:', context_length)
    print('Steps per epoch:', steps_per_epoch)
    
    activation_fn = nn.gelu
    config['activation_fn'] = 'gelu'
    dtype = jnp.bfloat16
    config['dtype'] = 'bf16'
    model = DiffusionTransformer(
        attention_dim=config['attention_dim'],
        num_attention_heads=config['num_attention_heads'],
        embedding_dim=config['embedding_dim'],
        feed_forward_dim=config['feed_forward_dim'],
        num_blocks=config['num_blocks'],
        token_dim=token_dim,
        context_length=context_length,
        activation_fn=activation_fn,
        dtype=dtype,
        remat=config['remat']
    )
    x = jnp.ones((batch_size, context_length, token_dim))
    t = jnp.ones((batch_size, 1))
    params = model.init(jax.random.PRNGKey(388), x, t)['params']
    tx = optax.chain(
        optax.zero_nans(),
        optax.adaptive_grad_clip(clipping=config['grad_clip']),
        optax.adamw(
            learning_rate=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    config['param_count'] = param_count
    print('Param count:', param_count)

    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(
        model=field_model, learning_rate=1e-3, KEY=jax.random.PRNGKey(2)
    )

    wandb.init(project='if-dit-r', config=config)
    train_loop(
        state=state, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch, 
        data_iterator=data_iterator, batch_size=batch_size, context_length=context_length, 
        token_dim=token_dim, output_dir=output_dir, field_state=field_state,
        image_width=image_width, image_height=image_height, field_param_map=field_param_map
    )

if __name__ == '__main__':
    main()
