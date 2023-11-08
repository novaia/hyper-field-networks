# Sliding Window Linear Transform DDIM.
from typing import Any, List, Tuple, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
import math
import json
from functools import partial
from operator import itemgetter
from dataclasses import dataclass
from hypernets.common.nn import SinusoidalEmbedding, LinearTransformer
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion
from hypernets.packing.ngp_nerf import unpack_weights, pack_weights, generate_weight_map
from fields import ngp_nerf
from fields.common.dataset import load_nerf_dataset

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
        print('Base sample shape:', base_sample.shape)
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

def sliding_reverse_diffusion(
    apply_fn:Callable, 
    params:Any,
    dataset_info:DatasetInfo,
    num_images:int, 
    diffusion_steps:int, 
    diffusion_schedule_fn:Callable,
    min_signal_rate:float,
    max_signal_rate:float,
    seed:int, 
    initial_noise:jax.Array = None,
):
    pred_weights = []
    for i in range(dataset_info.num_contexts):
        if initial_noise == None:
            initial_noise = jax.random.normal(
                jax.random.PRNGKey(seed), 
                shape=(num_images, dataset_info.context_length, dataset_info.token_dim)
            )
        step_size = 1.0 / diffusion_steps
        
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            
            diffusion_times = jnp.ones((num_images, 1, 1)) - step * step_size
            noise_rates, signal_rates = diffusion_schedule_fn(
                diffusion_times, min_signal_rate, max_signal_rate
            )
            pred_noises = jax.lax.stop_gradient(
                apply_fn(
                    {'params': params}, 
                    [noisy_images, noise_rates**2, jnp.full((num_images, 1, 1), i)], 
                )
            )
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = diffusion_schedule_fn(
                next_diffusion_times, min_signal_rate, max_signal_rate
            )
            next_noisy_images = (next_signal_rates*pred_images + next_noise_rates*pred_noises)
        pred_weights.append(pred_images)
    pred_weights = jnp.concatenate(pred_weights, axis=1)
    pred_weights = pred_weights[:, dataset_info.num_pad_tokens:, :]
    return pred_weights

def save_image(image, file_path):
    image = image - jnp.min(image)
    image = image / jnp.max(image)
    plt.imsave(file_path, image)

def render_from_generated_weights(
    sensor_dataset_path:str, param_map_path:str, packed_weights:jax.Array
):
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = True

    batch_size = 256 * 1024
    scene_bound = 1.0
    grid_resolution = 128
    diagonal_n_steps = 1024
    stepsize_portion = 1.0 / 256.0

    model = ngp_nerf.NGPNerf(
        number_of_grid_levels=num_hash_table_levels,
        max_hash_table_entries=max_hash_table_entries,
        hash_table_feature_dim=hash_table_feature_dim,
        coarsest_resolution=coarsest_resolution,
        finest_resolution=finest_resolution,
        density_mlp_width=density_mlp_width,
        color_mlp_width=color_mlp_width,
        high_dynamic_range=high_dynamic_range,
        exponential_density_activation=exponential_density_activation,
        scene_bound=scene_bound
    )
    KEY = jax.random.PRNGKey(0)
    KEY, state_init_key = jax.random.split(KEY)
    state = ngp_nerf.create_train_state(model, state_init_key, 1, 1, 1)

    with open(param_map_path, 'r') as f:
        param_map = json.load(f)
    unpacked_weights = unpack_weights(packed_weights, param_map)
    state = state.replace(params=unpacked_weights)
    
    occupancy_grid = ngp_nerf.create_occupancy_grid(
        resolution=grid_resolution, 
        update_interval=1, 
        warmup_steps=1
    )
    KEY, occupancy_update_key = jax.random.split(KEY)
    occupancy_grid.densities = ngp_nerf.update_occupancy_grid_density(
        KEY=occupancy_update_key,
        batch_size=batch_size,
        densities=occupancy_grid.densities,
        occupancy_mask=occupancy_grid.mask,
        grid_resolution=occupancy_grid.resolution,
        num_grid_entries=occupancy_grid.num_entries,
        scene_bound=scene_bound,
        state=state,
        warmup=True
    )
    occupancy_grid.mask, occupancy_grid.bitfield = ngp_nerf.threshold_occupancy_grid(
        diagonal_n_steps=diagonal_n_steps,
        scene_bound=scene_bound,
        densities=occupancy_grid.densities
    )

    dataset = load_nerf_dataset(sensor_dataset_path, downscale_factor=1)
    render_fn = partial(
        ngp_nerf.render_scene,
        patch_size_x=32,
        patch_size_y=32,
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        grid_cascades=1,
        grid_resolution=grid_resolution,
        stepsize_portion=stepsize_portion,
        occupancy_bitfield=occupancy_grid.bitfield,
        batch_size=batch_size,
        state=state
    )
    ngp_nerf.turntable_render(
        num_frames=3,
        camera_distance=2,
        render_fn=render_fn,
        file_name='generated_render'
    )

def main():
    normal_dtype = jnp.float32
    quantized_dtype = jnp.float16
    batch_size = 1
    dataset_path = 'data/synthetic_nerfs/packed_aliens'
    context_length = 150_000

    dataset = create_dataset_info(dataset_path, context_length, batch_size, verbose=True)
    token_dim = dataset.token_dim

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
        print('Loss:', loss)
    print('Finished training')

    pred_weights = sliding_reverse_diffusion(
        apply_fn=state.apply_fn,
        params=state.params,
        dataset_info=dataset,
        num_images=1,
        diffusion_steps=20,
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=0.02,
        max_signal_rate=0.95,
        seed=0,
    )
    print('Pred weights:', pred_weights.shape)
    render_from_generated_weights(
        sensor_dataset_path='data/synthetic_nerf_data/aliens/alien_2',
        param_map_path=os.path.join(dataset_path, 'param_map.json'),
        packed_weights=pred_weights[0]
    )

if __name__ == '__main__':
    main()