import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from fields import ngp_nerf, Dataset
from volrendjax import integrate_rays, integrate_rays_inference, march_rays, march_rays_inference
from volrendjax import morton3d_invert, packbits
from dataclasses import dataclass
import dataclasses
from typing import Callable, List, Literal, Tuple, Type, Union, Optional
from flax import struct
import numpy as np
from functools import partial
import optax
import json
from PIL import Image
import matplotlib.pyplot as plt
from fields import Dataset
import os

@dataclass
class OccupancyGrid:
    grid_resolution: int
    num_entries: int
    densities: jax.Array # Full precision density values.
    mask: jax.Array # Non-compact boolean representation of occupancy.
    bitfield: jax.Array # Compact occupancy bitfield.

def create_occupancy_grid(grid_resolution: int=128) -> OccupancyGrid:
    num_entries = grid_resolution**3
    # Each bit is an occupancy value, and uint8 is 8 bytes, so divide num_entries by 8.
    # This gives one entry per bit.
    bitfield = 255 * jnp.ones(shape=(num_entries // 8,), dtype=jnp.uint8)
    densities = jnp.zeros(shape=(num_entries,), dtype=jnp.float32)
    mask = jnp.zeros(shape=(num_entries,), dtype=jnp.bool_)
    return OccupancyGrid(grid_resolution, num_entries, densities, mask, bitfield)

def create_train_state(
    model:nn.Module, 
    rng,
    learning_rate:float, 
    epsilon:float, 
    weight_decay_coefficient:float
):
    x = (jnp.ones([3]) / 3, jnp.ones([3]) / 3)
    variables = model.init(rng, x)
    params = variables['params']
    adam = optax.adam(learning_rate, eps=epsilon, eps_root=epsilon)
    # To prevent divergence after long training periods, the paper applies a weak 
    # L2 regularization to the network weights, but not the hash table entries.
    weight_decay_mask = dict({
        key:True if key != 'MultiResolutionHashEncoding_0' else False
        for key in params.keys()
    })
    weight_decay = optax.add_decayed_weights(weight_decay_coefficient, mask=weight_decay_mask)
    tx = optax.chain(adam, weight_decay)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def process_3x4_transform_matrix(original:jnp.ndarray, scale:float):
    # Note that the translation component is not shifted.
    # This is different than the implementation in ngp_nerf (non-cuda).
    new = jnp.array([
        [original[1, 0], -original[1, 1], -original[1, 2], original[1, 3] * scale],
        [original[2, 0], -original[2, 1], -original[2, 2], original[2, 3] * scale],
        [original[0, 0], -original[0, 1], -original[0, 2], original[0, 3] * scale],
    ])
    return new

def load_dataset(dataset_path:str, downscale_factor:int):
    with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    frame_data = transforms['frames']
    first_file_path = frame_data[0]['file_path']
    # Process file paths if they're in the original nerf format.
    if not first_file_path.endswith('.png') and first_file_path.startswith('.'):
        process_file_path = lambda path: path[2:] + '.png'
    else:
        process_file_path = lambda path: path

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        file_path = process_file_path(frame['file_path'])
        image = Image.open(os.path.join(dataset_path, file_path))
        image = image.resize(
            (image.width // downscale_factor, image.height // downscale_factor),
            resample=Image.NEAREST
        )
        images.append(jnp.array(image))

    transform_matrices = jnp.array(transform_matrices)[:, :3, :]
    mean_translation = jnp.mean(jnp.linalg.norm(transform_matrices[:, :, -1], axis=-1))
    translation_scale = 1 / mean_translation
    process_transform_matrices_vmap = jax.vmap(process_3x4_transform_matrix, in_axes=(0, None))
    transform_matrices = process_transform_matrices_vmap(transform_matrices, translation_scale)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    dataset = Dataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_x'],
        fl_x=1,
        fl_y=1,
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        aabb_scale=1,
        transform_matrices=transform_matrices,
        images=images
    )
    dataset.fl_x = float(dataset.cx / jnp.tan(dataset.horizontal_fov / 2))
    dataset.fl_y = float(dataset.cy / jnp.tan(dataset.vertical_fov / 2))
    return dataset