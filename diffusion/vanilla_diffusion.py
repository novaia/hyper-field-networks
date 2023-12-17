from typing import Any, Callable, List
import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
from jax import lax
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
from datetime import datetime
import math
import orbax.checkpoint as ocp

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
        images = fn.cast(images, dtype=types.FLOAT)
        images = (images / types.Constant(127.5).float32()) - types.Constant(1.0).float32()
        return images
    
    train_pipeline = my_pipeline_def(
        source=external_iterator, 
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=0
    )
    train_iterator = DALIGenericIterator(pipelines=[train_pipeline], output_map=['x'])
    return train_iterator, steps_per_epoch

def precompute_patch_data(image_size, patch_size):
    num_patches_across = image_size // patch_size
    num_patches_total = num_patches_across**2
    indices = jnp.arange(num_patches_across) * patch_size
    horizontal_indices, vertical_indices = jnp.meshgrid(indices, indices)
    return num_patches_across, num_patches_total, horizontal_indices, vertical_indices

@partial(jax.jit, static_argnames=['num_patches_total', 'patch_size', 'channels'])
def patchify(x, num_patches_total, horizontal_indices, vertical_indices, patch_size, channels):
    def get_patch(x, horizontal_id, vertical_id):
        start_indices = [horizontal_id, vertical_id, 0]
        slice_sizes = [patch_size, patch_size, channels]
        patch = lax.dynamic_slice(x, start_indices, slice_sizes)
        return jnp.ravel(patch)

    patches = jax.vmap(
        jax.vmap(get_patch, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
    )(x, horizontal_indices, vertical_indices)
    patches = jnp.reshape(patches, (num_patches_total, patch_size*patch_size*channels))
    return patches

@partial(jax.jit, static_argnames=[
    'patch_size', 'num_patches_across', 'num_patches_total', 'channels'
])
def depatchify(x, patch_size, num_patches_across, num_patches_total, channels):
    x = jnp.reshape(x, (num_patches_total, patch_size, patch_size, channels))
    x = jnp.array(jnp.split(x, num_patches_across, axis=0))
    x = jax.vmap(lambda a: jnp.concatenate(a, axis=0), in_axes=0)(x)
    x = jnp.concatenate(x, axis=1)
    return x

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

class ImageSelfAttention(nn.Module):
    head_dim: int
    num_heads: int
    patch_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.head_dim == 0:
            return x
        input_features = x.shape[-1]
        num_patches_across, num_patches_total, horizontal_indices, vertical_indices = (
            precompute_patch_data(x.shape[1], self.patch_size)
        )
        x = jax.vmap(patchify, in_axes=(0, None, None, None, None, None))(
            x, 
            num_patches_total, 
            horizontal_indices,
            vertical_indices,
            self.patch_size,
            input_features
        )
        residual = x
        x = nn.LayerNorm(dtype=jnp.float32)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads*self.head_dim,
            out_features=x.shape[-1],
            dtype=jnp.float32
        )(inputs_q=x, inputs_kv=x)
        x = x + residual
        x = jax.vmap(depatchify, in_axes=(0, None, None, None, None))(
            x,
            self.patch_size,
            num_patches_across,
            num_patches_total,
            input_features
        )
        return x

class ResidualBlock(nn.Module):
    num_features: int
    num_groups: int
    patch_size: int
    head_dim: int
    num_heads: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb):
        input_features = x.shape[-1]
        if input_features == self.num_features:
            residual = x
        else:
            residual = nn.Conv(self.num_features, kernel_size=(1, 1), dtype=self.dtype)(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3), dtype=self.dtype)(x)
        x = nn.GroupNorm(self.num_groups, dtype=jnp.float32)(x)
        x = self.activation_fn(x)
        time_emb = nn.Dense(self.num_features, dtype=self.dtype)(time_emb)
        time_emb = self.activation_fn(time_emb)
        time_emb = jnp.broadcast_to(time_emb, x.shape)
        x = x + time_emb
        x = nn.Conv(self.num_features, kernel_size=(3, 3), dtype=self.dtype)(x)
        x = nn.GroupNorm(self.num_groups, dtype=jnp.float32)(x)
        x = self.activation_fn(x)
        x = x + residual
        x = ImageSelfAttention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            dtype=self.dtype
        )(x)
        return x

class DownBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    patch_size: int
    head_dim: int
    num_heads: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb, skips):
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features, 
                num_groups=self.num_groups,
                patch_size=self.patch_size,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x, time_emb)
            skips.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips

class UpBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    patch_size: int
    head_dim: int
    num_heads: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb, skips):
        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(
                num_features=self.num_features,
                num_groups=self.num_groups,
                patch_size=self.patch_size,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x, time_emb)
        return x, skips

class VanillaDiffusion(nn.Module):
    embedding_dim: int
    embedding_max_frequency: float
    num_features: List[int]
    num_groups: List[int]
    block_depth: int
    patch_sizes: List[int]
    head_dims: List[int]
    num_heads: List[int]
    output_channels: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, diffusion_time):
        time_emb = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_max_frequency=self.embedding_max_frequency,
            dtype=self.dtype
        )(diffusion_time)

        skips = []
        block_params = list(zip(
            self.num_features, 
            self.num_groups, 
            self.patch_sizes, 
            self.head_dims, 
            self.num_heads
        ))[:-1]
        for features, groups, patch, head_dim, heads in block_params:
            x, skips = DownBlock(
                num_features=features,
                num_groups=groups,
                block_depth=self.block_depth,
                patch_size=patch,
                head_dim=head_dim,
                num_heads=heads,
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x, time_emb, skips)
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features[-1],
                num_groups=self.num_groups[-1],
                patch_size=self.patch_sizes[-1],
                head_dim=self.head_dims[-1],
                num_heads=self.head_dims[-1],
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x, time_emb)
        for features, groups, patch, head_dim, heads in list(reversed(block_params)):
            x, skips = UpBlock(
                num_features=features,
                num_groups=groups,
                block_depth=self.block_depth,
                patch_size=patch,
                head_dim=head_dim,
                num_heads=heads,
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x, time_emb, skips)

        x = nn.Conv(self.output_channels, kernel_size=(1, 1), dtype=jnp.float32)(x)
        return x

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

@partial(jax.jit, static_argnames=['min_signal_rate', 'max_signal_rate', 'noise_clip'])
def train_step(state, images, min_signal_rate, max_signal_rate, noise_clip, key):
    noise_key, diffusion_time_key = jax.random.split(key, 2)
    noises = jax.random.normal(noise_key, images.shape, dtype=jnp.float32)
    # Clipping the noise prevents NaN loss when train_step is compiled on GPU, however this 
    # diverges from the math of typical diffusion models since the noise is no longer Gaussian. 
    # Standardizing the images to 0 mean and unit variance might have the same effect while 
    # remaining in line with standard practice. Setting NaN gradients to 0 also prevents NaN 
    # loss, but it feels kind of hacky and might obscure other numerical issues. 
    noises = jnp.clip(noises, -noise_clip, noise_clip)
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
    noise_clip:float,
    seed:int, 
):
    @jax.jit
    def inference_fn(state, noisy_images, diffusion_times):
        return jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_images, diffusion_times)
        )
    
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_images, image_height, image_width, channels)
    )
    initial_noise = jnp.clip(initial_noise, -noise_clip, noise_clip)
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = inference_fn(state, noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    return pred_images

def main():
    gpu = jax.devices('gpu')[0]
    print(gpu)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--wandb', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs_between_previews', type=int, default=1)
    parser.add_argument('--save_checkpoints', type=int, choices=[0, 1], default=1)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    args = parser.parse_args()

    if args.wandb == 1:
        import wandb

    # Config string maps.
    activation_fn_map = {'gelu': nn.gelu}
    dtype_map = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}

    with open(args.config, 'r') as f:
        config = json.load(f)
    assert len(config['num_features']) == len(config['num_groups']), (
        'len(num_features) must equal len(num_groups).'
    )
    activation_fn_name = config['activation_fn']
    assert activation_fn_name in activation_fn_map.keys(), (
        f'Invalid activation function: {activation_fn_name}. ',
        f'Must be one of the following: {activation_fn_map.keys()}.'
    )
    activation_fn = activation_fn_map[activation_fn_name]
    dtype_name = config['dtype']
    assert dtype_name in dtype_map.keys(), (
        f'Invalid dtype: {dtype_name}. Must be one of the following: {dtype_map.keys()}'
    )
    dtype = dtype_map[dtype_name]

    model = VanillaDiffusion(
        embedding_dim=config['embedding_dim'],
        embedding_max_frequency=config['embedding_max_frequency'],
        num_features=config['num_features'],
        num_groups=config['num_groups'],
        block_depth=config['block_depth'],
        patch_sizes=config['patch_sizes'],
        head_dims=config['head_dims'],
        num_heads=config['num_heads'],
        output_channels=config['output_channels'],
        activation_fn=activation_fn,
        dtype=dtype
    )
    x = jnp.ones(
        (
            config['batch_size'], 
            config['image_size'], 
            config['image_size'], 
            config['output_channels']
        ),
        dtype=dtype
    )
    diffusion_times = jnp.ones((config['batch_size'], 1, 1, 1), dtype=dtype)
    params = model.init(jax.random.PRNGKey(0), x, diffusion_times)['params']
    tx = optax.adam(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    config['param_count'] = param_count
    print('Param count', param_count)

    data_iterator, steps_per_epoch = get_data_iterator(args.dataset, config['batch_size'])

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.checkpoint_path is not None:
        state = checkpointer.restore(args.checkpoint_path, item=state)

    if args.wandb == 1: 
        wandb.init(project='vanilla-diffusion', config=config)

    print('Steps per epoch', steps_per_epoch)
    min_signal_rate = config['min_signal_rate']
    max_signal_rate = config['max_signal_rate']
    noise_clip = config['noise_clip']
    for epoch in range(config['epochs']):
        epoch_start_time = datetime.now()
        for _ in range(steps_per_epoch):
            images = next(data_iterator)['x']
            images = jnp.array(images, dtype=jnp.float32)
            images = jax.device_put(images, gpu)
            step_key = jax.random.PRNGKey(state.step)
            loss, state = train_step(
                state, images, min_signal_rate, max_signal_rate, noise_clip, step_key
            )
            if args.wandb == 1: 
                wandb.log({'loss': loss}, step=state.step)
            else:
                print(state.step, loss)
        epoch_end_time = datetime.now()
        print(
            f'Epoch {epoch} completed in {epoch_end_time-epoch_start_time} at {epoch_end_time}'
        )

        if args.save_checkpoints == 1:
            checkpointer.save(
                f'data/vanilla_diffusion_checkpoints/vd_step_{state.step}', state, force=True
            )
        if epoch % args.epochs_between_previews != 0:
            continue

        generated_images = reverse_diffusion(
            state=state, 
            num_images=8,
            diffusion_steps=1000,
            image_width=config['image_size'],
            image_height=config['image_size'],
            channels=config['output_channels'],
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            noise_clip=noise_clip,
            seed=epoch
        )
        generated_images = ((generated_images + 1.0) / 2.0) * 255.0
        generated_images = jnp.clip(generated_images, 0.0, 255.0)
        generated_images = np.array(generated_images, dtype=np.uint8)
        for i in range(generated_images.shape[0]):
            image = Image.fromarray(generated_images[i])
            image.save(f'data/vanilla_diffusion_output/step{state.step}_image{i}.png')

if __name__ == '__main__':
    main()