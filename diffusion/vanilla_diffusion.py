from typing import Any, Callable, List
import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
import argparse
import json
from functools import partial
import optax
import numpy as np
from random import shuffle
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
import nvidia.dali.types as types
import glob
from PIL import Image
import os
import time
import math

class ExternalInputIterator(object):
    def __init__(self, paths, batch_size):
        self.batch_size = batch_size
        self.paths = paths
        shuffle(self.paths)

    def __iter__(self):
        self.i = 0
        self.n = len(self.paths)
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            jpeg_path = self.paths[self.i]
            with open(jpeg_path, 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return batch

def get_data_iterator(dataset_path, batch_size, num_threads=3):
    abs_dataset_path = os.path.abspath(dataset_path)
    paths = glob.glob(f'{abs_dataset_path}/*/*')
    shuffle(paths)
    steps_per_epoch = len(paths) // batch_size
    external_iterator = ExternalInputIterator(paths, batch_size)

    @pipeline_def
    def my_pipeline_def(source):
        jpegs = fn.external_source(
            source=source, 
            dtype=types.UINT8,
        )
        images = fn.decoders.image(jpegs, device='cpu', output_type=types.RGB)
        return images
    
    train_pipeline = my_pipeline_def(
        source=external_iterator, 
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=0
    )
    train_iterator = DALIGenericIterator(pipelines=[train_pipeline], output_map=['x'])
    return train_iterator, steps_per_epoch

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
    embedding_min_frequency:float = 1.0
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
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis=-1,
            dtype=self.dtype
        )
        return embeddings

class ResidualBlock(nn.Module):
    num_features: int
    num_groups: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb):
        input_features = x.shape[-1]
        if input_features == self.num_features:
            residual = x
        else:
            residual = nn.Conv(self.num_features, kernel_size=(1, 1))(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        time_emb = nn.Dense(self.num_features)(time_emb)
        time_emb = self.activation_fn(time_emb)
        time_emb = jnp.broadcast_to(time_emb, x.shape)
        x = x + time_emb
        x = nn.Conv(self.num_features, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features, 
                num_groups=self.num_groups,
                activation_fn=self.activation_fn
            )(x, time_emb)
            skips.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips

class UpBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(
                num_features=self.num_features,
                num_groups=self.num_groups,
                activation_fn=self.activation_fn
            )(x, time_emb)
        return x, skips

class VanillaDiffusion(nn.Module):
    embedding_dim: int
    embedding_max_frequency: float
    num_features: List[int]
    num_groups: List[int]
    block_depth: int
    output_channels: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, diffusion_time):
        time_emb = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_max_frequency=self.embedding_max_frequency
        )(diffusion_time)

        skips = []
        features_and_groups = list(zip(self.num_features[:-1], self.num_groups[:-1]))
        for f, g in features_and_groups:
            x, skips = DownBlock(
                num_features=f,
                num_groups=g,
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features[-1],
                num_groups=self.num_groups[-1],
                activation_fn=self.activation_fn
            )(x, time_emb)
        for f, g in list(reversed(features_and_groups)):
            x, skips = UpBlock(
                num_features=f,
                num_groups=g,
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)

        x = nn.Conv(
            self.output_channels, 
            kernel_size=(1, 1), 
        )(x)
        return x

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

@partial(jax.jit, static_argnames=['min_signal_rate', 'max_signal_rate'])
def train_step(state, images, min_signal_rate, max_signal_rate, key):
    noise_key, diffusion_time_key = jax.random.split(key, 2)
    noises = jax.random.normal(noise_key, images.shape, dtype=jnp.float32)
    # Clipping the noise prevents NaN loss when train_step is compiled on GPU, however this 
    # diverges from the math of typical diffusion models since the noise is no longer Gaussian. 
    # Standardizing the images to 0 mean and unit variance might have the same effect while 
    # remaining in line with standard practice. Setting NaN gradients to 0 also prevents NaN 
    # loss, but it feels kind of hacky and might obscure other numerical issues. 
    noises = jnp.clip(noises, -1.0, 1.0)
    diffusion_times = jax.random.uniform(diffusion_time_key, (images.shape[0], 1, 1, 1))
    noise_rates, signal_rates = diffusion_schedule(
        diffusion_times, min_signal_rate, max_signal_rate
    )
    noisy_images = signal_rates * images + noise_rates * noises

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_images, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    #grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
    state = state.apply_gradients(grads=grads)
    return loss, state

#@partial(jax.jit, static_argnames=[
#    'num_images', 'diffusion_steps', 'image_width', 'image_height', 'channels', 
#    'min_signal_rate', 'max_signal_rate'
#])
def reverse_diffusion(
    state, 
    num_images:int, 
    diffusion_steps:int, 
    image_width:int, 
    image_height:int, 
    channels:int,
    min_signal_rate:float,
    max_signal_rate:float,
    seed:int, 
):
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_images, image_height, image_width, channels)
    )
    initial_noise = jnp.clip(initial_noise, -1.0, 1.0)
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_images, noise_rates**2)
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    return pred_images

def main():
    wandb_logging = False
    if wandb_logging:
        import wandb
    gpu = jax.devices('gpu')[0]
    print(gpu)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    assert len(config['num_features']) == len(config['num_groups']), (
        'len(num_features) must equal len(num_groups).'
    )
    activation_fn_name = config['activation_fn']
    activation_fn_map = {'gelu': nn.gelu}
    assert activation_fn_name in activation_fn_map.keys(), (
        f'Invalid activation function: {activation_fn_name}. ',
        f'Must be one of the following: {activation_fn_map.keys()}.'
    )
    activation_fn = activation_fn_map[activation_fn_name]

    model = VanillaDiffusion(
        embedding_dim=config['embedding_dim'],
        embedding_max_frequency=config['embedding_max_frequency'],
        num_features=config['num_features'],
        num_groups=config['num_groups'],
        block_depth=config['block_depth'],
        output_channels=config['output_channels'],
        activation_fn=activation_fn
    )
    x = jnp.ones(
        (
            config['batch_size'], 
            config['image_size'], 
            config['image_size'], 
            config['output_channels']
        ),
        dtype=jnp.float32
    )
    diffusion_times = jnp.ones((config['batch_size'], 1, 1, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x, diffusion_times)['params']
    tx = optax.adam(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    data_iterator, steps_per_epoch = get_data_iterator(args.dataset, config['batch_size'])

    if wandb_logging: 
        wandb.init(project='vanilla-diffusion', config=config)

    print('Steps per epoch', steps_per_epoch)
    min_signal_rate = config['min_signal_rate']
    max_signal_rate = config['max_signal_rate']
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        for _ in range(steps_per_epoch):
            images = next(data_iterator)['x']
            images = jnp.array(images, dtype=jnp.float32)
            images = jax.device_put(images, gpu)
            images = ((images / 255.0) * 2.0) - 1.0
            step_key = jax.random.PRNGKey(state.step)
            loss, state = train_step(state, images, min_signal_rate, max_signal_rate, step_key)
            if wandb_logging: 
                wandb.log({'loss': loss}, step=state.step)
            else:
                print(state.step, loss)
        epoch_end_time = time.time()
        print(
            f'Epoch {epoch} completed in {epoch_end_time-epoch_start_time}'
        )

        generated_images = reverse_diffusion(
            state=state, 
            num_images=8,
            diffusion_steps=20,
            image_width=config['image_size'],
            image_height=config['image_size'],
            channels=config['output_channels'],
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            seed=epoch
        )
        generated_images = ((generated_images + 1.0) / 2.0) * 255.0
        generated_images = jnp.clip(generated_images, 0.0, 255.0)
        generated_images = np.array(generated_images, dtype=np.uint8)
        for i in range(generated_images.shape[0]):
            image = Image.fromarray(generated_images[i])
            image.save(f'data/vanilla_diffusion_output/epoch{epoch}_image{i}.png')

if __name__ == '__main__':
    main()