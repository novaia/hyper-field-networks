# The purpose of this script is to see whether or not conditioning an image DiT
# with labels (i.e. MNIST digit labels) significantly improves performance.
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from hypernets.common.nn import VanillaTransformer
from matplotlib import pyplot as plt
import wandb
import pandas as pd

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

class Mlp(nn.Module):
    dim: int
    depth: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(features=self.dim, dtype=self.dtype)(x)
            x = self.activation_fn(x)
        return x

class AdaLayerNorm(nn.Module):
    dim: int
    depth: int
    activation_fn: Callable
    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, ada_input):
        x = nn.LayerNorm(
            use_bias=False, use_scale=False, 
            epsilon=self.epsilon, dtype=self.dtype
        )(x)
        ada_params = Mlp(
            dim=self.dim, depth=self.depth, 
            activation_fn=self.activation_fn, dtype=self.dtype
        )(ada_input)
        ada_params = nn.Dense(features=2, dtype=self.dtype)(x)
        scale = ada_params[..., 0:1]
        bias = ada_params[..., 1:2]
        x = x * (1 + scale) + bias
        return x

class DiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    embedding_dim:int
    num_blocks:int
    feed_forward_dim:int
    token_dim:int
    context_length:int
    num_labels:int
    activation_fn:Callable
    dtype:Any
    remat:bool
    ada_ln_mlp_depth:int
    ada_ln_mlp_width:int

    @nn.compact
    def __call__(self, x, t, label):
        num_tokens = self.context_length + 2
        positions = jnp.reshape(jnp.arange(num_tokens), (1, num_tokens, 1))
        position_embedding = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            dtype=self.dtype
        )(positions)
        
        time_embedding = SinusoidalEmbedding(
            self.embedding_dim,
            dtype=self.dtype
        )(t)

        label_embedding = nn.Embed(
            num_embeddings=self.num_labels,
            features=self.embedding_dim
        )(jnp.ravel(label))

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
        
        time_embedding = jnp.expand_dims(time_embedding, axis=1)
        x = jnp.concatenate([x, time_embedding], axis=-2)
        # Add the label token to the end of the sequence.
        label_embedding = jnp.expand_dims(label_embedding, axis=1)
        x = jnp.concatenate([x, label_embedding], axis=-2)
        x = x + position_embedding
        
        for _ in range(self.num_blocks):
            residual = x
            #x = AdaLayerNorm(
            #    dim=self.ada_ln_mlp_width, 
            #    depth=self.ada_ln_mlp_depth,
            #    activation_fn=self.activation_fn,
            #    dtype=self.dtype
            #)(x, time_embedding)
            x = nn.RMSNorm()(x)
            Attention = nn.MultiHeadDotProductAttention
            if self.remat:
                Attention = nn.remat(Attention)
            x = Attention(
                num_heads=self.num_attention_heads,
                dtype=self.dtype,
                qkv_features=self.attention_dim,
                out_features=self.embedding_dim
            )(inputs_q=x, inputs_kv=x)
            x = x + residual
            residual = x
            #x = AdaLayerNorm(
            #    dim=self.ada_ln_mlp_width, 
            #    depth=self.ada_ln_mlp_depth,
            #    activation_fn=self.activation_fn,
            #    dtype=self.dtype
            #)(x, time_embedding)
            x = nn.RMSNorm()(x)
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = self.activation_fn(x)
            x = nn.Dense(features=self.embedding_dim)(x)
            x = x + residual

        # Remove the label token from the end of the sequence.
        x = x[:, :-2, :]
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
    labels,
    num_samples:int, 
    diffusion_steps:int, 
    token_dim:int, 
    context_length:int,
    seed:int 
):
    @jax.jit
    def inference_fn(state, noisy_images, diffusion_times, labels):
        return jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_images, diffusion_times, labels)
        )
    
    labels = jnp.expand_dims(jnp.array(labels), axis=-1)
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_samples, context_length, token_dim)
    )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_samples, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        pred_noises = inference_fn(state, noisy_images, noise_rates**2, labels)
        pred_images = (
            (noisy_images - jnp.expand_dims(noise_rates, axis=1) * pred_noises) 
            / jnp.expand_dims(signal_rates, axis=1)
        )
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(next_diffusion_times)
        next_noisy_images = (
            jnp.expand_dims(next_signal_rates, axis=1) * pred_images
            + jnp.expand_dims(next_noise_rates, axis=1) * pred_noises
        )
    return pred_images

@partial(jax.jit, static_argnames=['batch_size'])
def train_step(state, images, labels, batch_size, seed):
    noise_key, diffusion_time_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    noises = jax.random.normal(noise_key, images.shape, dtype=jnp.float32)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch_size, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times)
    noisy_images = (
        jnp.expand_dims(signal_rates, axis=1) * images 
        + jnp.expand_dims(noise_rates, axis=1) * noises
    )

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_images, noise_rates**2, labels)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def train_loop(
    state, num_epochs, steps_per_epoch, data_iterator, 
    batch_size, context_length, token_dim, output_dir,
    image_width, image_height
):
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for step in range(steps_per_epoch):
            batch = next(data_iterator)
            images = batch['images']
            labels = batch['labels']
            loss, state = train_step(
                state, images=images, labels=labels, batch_size=batch_size, seed=state.step
            )
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        #print(average_loss)
        wandb.log({'loss': average_loss}, state.step)
        num_samples = 10
        samples = ddim_sample(
            state=state,
            num_samples=num_samples,
            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            diffusion_steps=20,
            token_dim=token_dim,
            context_length=context_length,
            seed=0
        )
        samples = (samples + 1.0) / 2.0
        samples = jnp.clip(samples, 0.0, 1.0)
        samples = jnp.reshape(samples, [num_samples, image_width, image_height])
        for i, sample in enumerate(samples):
            plt.imsave(os.path.join(output_dir, f'epoch{epoch}_image{i}.png'), sample)

def get_data_iterator(batch_size, context_length, token_dim, dataset_path):
    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def wds_pipeline():
        raw_image, ascii_label = fn.readers.webdataset(
            paths=dataset_path, 
            ext=['png', 'cls'], 
            missing_component_behavior='error',
            random_shuffle=True
        )
        image = fn.decoders.image(raw_image, output_type=dali_types.GRAY).gpu()
        image = fn.cast(image, dtype=dali_types.FLOAT)
        image_scale = dali_types.Constant(127.5).float32()
        image_shift = dali_types.Constant(1.0).float32()
        image = (image / image_scale) - image_shift
        image = fn.reshape(image, shape=(context_length, token_dim))
        ascii_shift = dali_types.Constant(48).uint8()
        label = ascii_label.gpu() - ascii_shift
        return image, label

    data_pipeline = wds_pipeline()
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], 
        output_map=['images', 'labels'], 
        last_batch_policy=LastBatchPolicy.DROP
    )
    return data_iterator

def main():
    output_dir = 'data/dit_runs/10'
    config_path = 'configs/conditional_image_dit.json'
    dataset_path = 'data/mnist-webdataset-png/data.tar'
    num_epochs = 1000

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_labels = 10
    image_height = 28
    image_width = 28
    token_dim = config['token_dim']
    context_length = int((image_height * image_width) // token_dim)
    batch_size = config['batch_size']
    steps_per_epoch = int(60_000 // batch_size)

    data_iterator = get_data_iterator(
        batch_size=batch_size,
        context_length=context_length,
        token_dim=token_dim,
        dataset_path=dataset_path
    )
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
        num_labels=num_labels,
        activation_fn=activation_fn,
        dtype=dtype,
        remat=config['remat'],
        ada_ln_mlp_depth=config['ada_ln_mlp_depth'],
        ada_ln_mlp_width=config['ada_ln_mlp_width']
    )
    x = jnp.ones((batch_size, context_length, token_dim))
    t = jnp.ones((batch_size, 1))
    label = jnp.ones((batch_size, 1), dtype=jnp.uint8)
    params = model.init(jax.random.PRNGKey(388), x, t, label)['params']
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
    print(f'Param count: {param_count:,}')
    
    wandb.init(project='if-dit', config=config)
    train_loop(
        state=state, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch, 
        data_iterator=data_iterator, batch_size=batch_size, context_length=context_length, 
        token_dim=token_dim, output_dir=output_dir, 
        image_width=image_width, image_height=image_height
    )

if __name__ == '__main__':
    main()
