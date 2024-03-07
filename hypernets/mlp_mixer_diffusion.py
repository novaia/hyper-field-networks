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

class DiffusionMlpMixer(nn.Module):
    dim: int
    context_length: int
    output_dim: int
    num_labels: int
    num_blocks: int
    mlp_depth: int
    ada_ln_depth: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t, label):
        t_emb = SinusoidalEmbedding(embedding_dim=self.dim, dtype=self.dtype)(t)
        label_emb = nn.Embed(
            num_embeddings=self.num_labels,
            features=self.dim
        )(jnp.ravel(label))

        x = Mlp(
            dim=self.dim, depth=3, 
            activation_fn=self.activation_fn, dtype=self.dtype
        )(x)
        label_emb = jnp.expand_dims(label_emb, axis=1)
        x = jnp.concatenate([x, label_emb], axis=-2)
        num_tokens = self.context_length + 1

        def norm(x):
            return AdaLayerNorm(
                dim=self.dim, depth=self.ada_ln_depth, 
                activation_fn=self.activation_fn, dtype=self.dtype
            )(x, t_emb)

        def transpose(x):
            return jnp.swapaxes(x, -1, -2)

        def mlp(x, transposed):
            dim = self.dim if not transposed else num_tokens
            return Mlp(
                dim=dim, depth=self.mlp_depth, 
                activation_fn=self.activation_fn, dtype=self.dtype
            )(x)

        for _ in range(self.num_blocks):
            residual = x
            x = norm(x)
            x = transpose(x)
            x = mlp(x, transposed=True)
            x = transpose(x)
            x = x + residual
            residual = x
            x = norm(x)
            x = mlp(x, transposed=False)
            x = x + residual
        
        x = x[:, :-1, :]
        x = nn.Dense(
            features=self.output_dim, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
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
        print(f'Epoch {epoch}, loss: {average_loss}')
        #wandb.log({'loss': average_loss}, state.step)
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
    output_dir = 'data/mlp_mixer_diffusion_runs/1'
    dataset_path = 'data/mnist-webdataset-png/data.tar'
    num_epochs = 1000

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 32
    steps_per_epoch = 60_000 // batch_size
    print(f'Steps per epoch: {steps_per_epoch:,}')
    token_dim = 1
    image_width = 28
    image_height = 28
    context_length = int((image_width * image_height) // token_dim)

    data_iterator = get_data_iterator(
        batch_size=batch_size,
        context_length=context_length,
        token_dim=token_dim,
        dataset_path=dataset_path
    )

    model = DiffusionMlpMixer(
        dim=128,
        context_length=context_length,
        output_dim=token_dim,
        ada_ln_depth=1,
        num_labels=10,
        num_blocks=16,
        mlp_depth=3,
        activation_fn=nn.gelu,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length, 1))
    t = jnp.ones((batch_size, 1))
    labels = jnp.ones((batch_size, 1), dtype=jnp.uint8)
    params = model.init(jax.random.PRNGKey(121), x, t, labels)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adaptive_grad_clip(clipping=3.0),
        optax.adam(learning_rate=3e-4)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'Param count: {param_count:,}')
    train_loop(
        state=state, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch, 
        data_iterator=data_iterator, batch_size=batch_size, context_length=context_length, 
        token_dim=token_dim, output_dir=output_dir, 
        image_width=image_width, image_height=image_height
    )

if __name__ == '__main__':
    main()
