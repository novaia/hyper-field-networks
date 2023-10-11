# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
from jax.random import PRNGKeyArray
import optax
import json
import os
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
import numba

@dataclass
class Dataset:
    horizontal_fov: float
    vertical_fov: float
    fl_x: float # Focal length x.
    fl_y: float # Focal length y.
    k1: float # First radial distortion parameter.
    k2: float # Second radial distortion parameter.
    p1: float # Third radial distortion parameter.
    p2: float # Fourth radial distortion parameter.
    cx: float # Principal point x.
    cy: float # Principal point y.
    w: int # Image width.
    h: int # Image height.
    aabb_scale: int # Scale of scene bounding box.
    transform_matrices: Optional[jnp.ndarray] = None
    images: Optional[jnp.ndarray] = None

# Calculates the fourth order spherical harmonic encoding for the given direction.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
def fourth_order_sh_encoding(direction:jnp.ndarray):
    x = direction[0]
    y = direction[1]
    z = direction[2]

    xx = x**2
    yy = y**2
    zz = z**2

    components = jnp.array([
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.9461746957575601 * zz - 0.31539156525251999,
        1.0925484305920792 * x * z,
        0.5462742152960396 * (xx - yy),
        0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * x * y * z,
        0.4570457994644658 * y * (5 * zz - 1),
        0.3731763325901154 * z * (5 * zz - 3),
        0.4570457994644658 * x * (5 * zz - 1),
        1.445305721320277 * z * (xx - yy),
        0.5900435899266435 * x * (xx - 3 * yy)
    ])

    return components

def alpha_composite(foreground, background, alpha):
    return foreground + background * (1 - alpha)

def render_pixel(
    densities:jnp.ndarray, colors:jnp.ndarray, z_vals:jnp.ndarray, directions:jnp.ndarray
):  
    eps = 1e-10
    deltas = jnp.concatenate([
        z_vals[1:] - z_vals[:-1], 
        jnp.broadcast_to(1e10, z_vals[:1].shape)
    ], axis=-1)
    deltas = jnp.expand_dims(deltas, axis=-1)
    deltas = deltas * jnp.linalg.norm(directions, keepdims=True, axis=-1)
    alphas = 1.0 - jnp.exp(-densities * deltas)
    accum_prod = jnp.concatenate([
        jnp.ones_like(alphas[:1], alphas.dtype),
        jnp.cumprod(1.0 - alphas[:-1] + eps, axis=0)
    ], axis=0)
    weights = alphas * accum_prod
    rendered_color = jnp.sum(weights * colors, axis=0)
    accumulated_alpha = jnp.sum(weights, axis=0)
    return rendered_color, accumulated_alpha

@jax.jit
def get_ray(uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    origin = transform_matrix[:3, -1]
    return origin, direction

@numba.njit
def trace_ray(ray_origin, ray_direction, z_vals, num_ray_samples):    
    box_min = 0
    box_max = 1
    num_valid_samples = 0
    valid_z_vals = np.zeros((num_ray_samples,), dtype=np.float32)
    valid_samples = np.zeros((num_ray_samples, 3), dtype=np.float32)
    matching_directions = np.zeros((num_ray_samples, 3))
    for z in z_vals:
        current_sample = [
            ray_origin[0] + ray_direction[0] * z, 
            ray_origin[1] + ray_direction[1] * z, 
            ray_origin[2] + ray_direction[2] * z
        ]
        x_check = current_sample[0] >= box_min and current_sample[0] <= box_max
        y_check = current_sample[1] >= box_min and current_sample[1] <= box_max
        z_check = current_sample[2] >= box_min and current_sample[2] <= box_max
        if x_check and y_check and z_check:
            valid_z_vals[num_valid_samples] = z
            valid_samples[num_valid_samples] = current_sample
            matching_directions[num_valid_samples] = [
                ray_direction[0], ray_direction[1], ray_direction[2]
            ]
            num_valid_samples += 1

    valid_z_vals = valid_z_vals[:num_valid_samples]
    valid_samples = valid_samples[:num_valid_samples]
    matching_directions = matching_directions[:num_valid_samples]
    return valid_samples, matching_directions, valid_z_vals, num_valid_samples

@numba.njit
def batch_trace_rays(ray_origins, ray_directions, z_vals, batch_size, num_ray_samples):
    batch_samples = np.zeros((batch_size * num_ray_samples, 3), dtype=np.float32)
    batch_directions = np.zeros((batch_size * num_ray_samples, 3), dtype=np.float32)
    batch_z_vals = np.zeros((batch_size * num_ray_samples,))
    ray_start_indices = np.zeros((batch_size,), dtype=np.int32)
    num_samples = 0
    for i in range(batch_size):
        new_samples, matching_directions, new_z_vals, num_new_samples = trace_ray(
            ray_origins[i],
            ray_directions[i],
            z_vals,
            num_ray_samples
        )
        ray_start_indices[i] = num_samples
        updated_num_samples = num_samples + num_new_samples
        batch_z_vals[num_samples:updated_num_samples] = new_z_vals
        batch_samples[num_samples:updated_num_samples] = new_samples
        batch_directions[num_samples:updated_num_samples] = matching_directions
        num_samples = updated_num_samples
    return batch_samples, batch_directions, batch_z_vals, ray_start_indices, num_samples

@numba.njit
def batch_render(densities, colors, z_vals, ray_start_indices, batch_size):
    rendered_colors = np.zeros((batch_size, 3), dtype=np.float32)
    for i in range(batch_size):
        start_index = ray_start_indices[i]
        end_index = ray_start_indices[i+1] if i < batch_size - 1 else ray_start_indices[-1]
        slice_size = end_index - start_index
        if start_index == end_index:
            continue

        current_densities = np.ravel(densities[start_index:end_index])
        current_colors = colors[start_index:end_index]
        current_z_vals = z_vals[start_index:end_index]

        deltas = np.zeros((slice_size,), dtype=np.float32)
        deltas[:-1] = current_z_vals[1:] - current_z_vals[:-1]
        deltas[-1] = 1e10
        alphas = 1.0 - np.exp(-current_densities * deltas)
        accum_prod = np.zeros((slice_size,), dtype=np.float32)
        accum_prod[0] = 1.0
        accum_prod[1:] = np.cumprod(1.0 - alphas[:-1] + 1e-10)
        weights = alphas * accum_prod
        weights = np.expand_dims(weights, axis=-1)
        rendered_colors[i] = np.sum(weights * current_colors, axis=0)
    return rendered_colors

class MultiResolutionHashEncoding(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.spatial_dim = 3
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        self.scalings = jnp.reshape(self.scalings, (self.scalings.shape[0], 1))
        absolute_table_size = self.table_size * self.num_levels
        # Feature dim comes first so features can be broadcast with point offset.
        # Feature shape is (feature_dim, num_levels). 
        # Point offset shape is (spatial_dim, num_levels).
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (self.feature_dim, absolute_table_size)
        )

    def hash_function(self, x:jnp.ndarray, table_size:int, hash_offset:jnp.ndarray):
        pre_xor = x * jnp.array([1, 2654435761, 805459861])
        x = jnp.bitwise_xor(pre_xor[:, 0], pre_xor[:, 1])
        x = jnp.bitwise_xor(x, pre_xor[:, 2])
        x = x % table_size
        x = x + hash_offset
        return x
    
    def __call__(self, x:jnp.ndarray):
        scaled = x * self.scalings
        scaled_c = jnp.ceil(scaled).astype(jnp.int32)
        scaled_f = jnp.floor(scaled).astype(jnp.int32)
        point_offset = jnp.reshape(scaled - scaled_f, (self.spatial_dim, self.num_levels))
        
        vertex_0 = scaled_c
        vertex_1 = jnp.concatenate([scaled_c[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_2 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_3 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_4 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_5 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_6 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_7 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)

        hashed_0 = self.hash_function(vertex_0, self.table_size, self.hash_offset)
        hashed_1 = self.hash_function(vertex_1, self.table_size, self.hash_offset)
        hashed_2 = self.hash_function(vertex_2, self.table_size, self.hash_offset)
        hashed_3 = self.hash_function(vertex_3, self.table_size, self.hash_offset)
        hashed_4 = self.hash_function(vertex_4, self.table_size, self.hash_offset)
        hashed_5 = self.hash_function(vertex_5, self.table_size, self.hash_offset)
        hashed_6 = self.hash_function(vertex_6, self.table_size, self.hash_offset)
        hashed_7 = self.hash_function(vertex_7, self.table_size, self.hash_offset)

        f_0 = self.hash_table[:, hashed_0]
        f_1 = self.hash_table[:, hashed_1]
        f_2 = self.hash_table[:, hashed_2]
        f_3 = self.hash_table[:, hashed_3]
        f_4 = self.hash_table[:, hashed_4]
        f_5 = self.hash_table[:, hashed_5]
        f_6 = self.hash_table[:, hashed_6]
        f_7 = self.hash_table[:, hashed_7]

        # Linearly interpolate between all of the features.
        f_03 = f_0 * point_offset[0:1, :] + f_3 * (1 - point_offset[0:1, :])
        f_12 = f_1 * point_offset[0:1, :] + f_2 * (1 - point_offset[0:1, :])
        f_56 = f_5 * point_offset[0:1, :] + f_6 * (1 - point_offset[0:1, :])
        f_47 = f_4 * point_offset[0:1, :] + f_7 * (1 - point_offset[0:1, :])

        f0312 = f_03 * point_offset[1:2, :] + f_12 * (1 - point_offset[1:2, :])
        f4756 = f_47 * point_offset[1:2, :] + f_56 * (1 - point_offset[1:2, :])

        encoded_value = f0312 * point_offset[2:3, :] + f4756 * (
            1 - point_offset[2:3, :]
        )
        # Transpose so that features are contiguous.
        # i.e. [[f_0_x, f_0_y], [f_1_x, f_2_y], ...]
        # Then ravel to get the entire encoding.
        return jnp.ravel(jnp.transpose(encoded_value))

class InstantNerf(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.
    density_mlp_width: int
    color_mlp_width: int
    high_dynamic_range: bool
    exponential_density_activation: bool

    @nn.compact
    def __call__(self, x):
        position, direction = x
        encoded_position = MultiResolutionHashEncoding(
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim
        )(position)
        x = encoded_position

        x = nn.Dense(self.density_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(16+1)(x)
        if self.exponential_density_activation:
            density = jnp.exp(x[0:1])
        else:
            density = nn.activation.relu(x[0:1])
        density_feature = x[1:]

        encoded_direction = fourth_order_sh_encoding(direction)
        x = jnp.concatenate([density_feature, jnp.ravel(encoded_direction)], axis=0)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(3)(x)

        if self.high_dynamic_range:
            color = jnp.exp(x)
        else:
            color = nn.activation.sigmoid(x)

        return density, color

def create_train_state(model:nn.Module, rng:PRNGKeyArray, learning_rate:float, epsilon:float):
    x = (jnp.ones([3]) / 3, jnp.ones([3]) / 3)
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate, eps=epsilon)
    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return ts

def sample_pixels(
    rng,
    num_samples:int, 
    image_width:int, 
    image_height:int, 
    num_images:int, 
    images:jnp.ndarray,
):
    width_rng, height_rng, image_rng = jax.random.split(rng, num=3) 
    width_indices = jax.random.randint(
        width_rng, shape=(num_samples,), minval=0, maxval=image_width
    )
    height_indices = jax.random.randint(
        height_rng, shape=(num_samples,), minval=0, maxval=image_height
    )
    image_indices = jax.random.randint(
        image_rng, shape=(num_samples,), minval=0, maxval=num_images
    )
    pixel_samples = images[image_indices, width_indices, height_indices]
    indices = (image_indices, width_indices, height_indices)
    return pixel_samples, indices 

def train_loop(
    batch_size:int, 
    num_ray_samples:int,
    ray_near:float,
    ray_far:float, 
    training_steps:int, 
    state:TrainState, 
    dataset:Dataset
):
    @jax.jit
    def compute_sample(params, ray_sample, direction):
        return state.apply_fn({'params': params}, (ray_sample, direction))
    compute_batch = jax.vmap(compute_sample, in_axes=(None, 0, 0))

    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    t_vals = jnp.linspace(0., 1., num_ray_samples)
    z_vals = ray_near * (1. - t_vals) + ray_far * t_vals
    cpus = jax.devices("cpu")
    for step in range(training_steps):
        pixel_sample_key, random_bg_key = jax.random.split(jax.random.PRNGKey(step), num=2)
        pixels, indices = sample_pixels(
            pixel_sample_key,
            batch_size,
            dataset.w,
            dataset.h,
            dataset.images.shape[0],
            dataset.images
        )
        image_indices, width_indices, height_indices = indices
        ray_origins, ray_directions = get_ray_vmap(
            width_indices, 
            height_indices, 
            dataset.transform_matrices[image_indices], 
            dataset.cx, 
            dataset.cy, 
            dataset.fl_x, 
            dataset.fl_y
        )
        
        cpu_ray_origins = jax.device_put(ray_origins, cpus[0])
        cpu_ray_directions = jax.device_put(ray_directions, cpus[0])
        cpu_z_vals = jax.device_put(z_vals, cpus[0])
        batch_samples, batch_directions, batch_z_vals, ray_start_indices, num_valid_samples = batch_trace_rays(
            cpu_ray_origins, cpu_ray_directions, cpu_z_vals, batch_size, num_ray_samples
        )
        batch_samples = jnp.array(batch_samples)
        batch_directions = jnp.array(batch_directions)
        batch_directions_norms = jnp.linalg.norm(batch_directions, keepdims=True, axis=-1)
        normalized_batch_directions = batch_directions / batch_directions_norms
        batch_z_vals = jnp.array(batch_z_vals)
        ray_start_indices = jnp.array(ray_start_indices)

        #print('Batch samples shape:', batch_samples.shape)
        #print('Batch directions shape:', batch_directions.shape)
        #print('Normalized batch directions shape:', normalized_batch_directions.shape)
        #print('Batch z vals shape:', batch_z_vals.shape)
        #print('Ray start indices shape:', ray_start_indices.shape)
        #print(ray_start_indices[:20])
        #print(jnp.diff(ray_start_indices[:20]))

        @jax.custom_vjp
        def differentiable_render(densities, colors):
            cpu_densities = np.array(jax.device_put(densities, cpus[0]))
            cpu_colors = np.array(jax.device_put(colors, cpus[0]))
            cpu_ray_start_indices = np.array(jax.device_put(ray_start_indices, cpus[0]))
            cpu_z_vals = np.array(jax.device_put(batch_z_vals, cpus[0]))
            rendered_colors = jnp.array(batch_render(
                cpu_densities, cpu_colors, cpu_z_vals, cpu_ray_start_indices, batch_size
            ))
            return rendered_colors
        
        def differentiable_render_fwd(densities, colors):
            return (
                differentiable_render(densities, colors), 
                (densities, colors) # Placeholder residuals.
            )
        
        # Placeholder backward pass.
        def differentiable_render_bwd(res, g):
            densities, colors = res
            return densities, colors
        
        differentiable_render.defvjp(differentiable_render_fwd, differentiable_render_bwd)

        def loss_fn(params):
            densities, colors = compute_batch(
                params, batch_samples, normalized_batch_directions
            )
            target_colors = pixels[:, :3]
            rendered_colors = differentiable_render(densities, colors)
            loss = jnp.mean((rendered_colors - target_colors)**2)
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        #print('Loss:', loss)
        #print('Total samples:', num_valid_samples)
        #print('Samples per ray:', num_valid_samples / batch_size)
    return state

def render_scene(
    num_ray_samples:int,
    patch_size_x:int,
    patch_size_y:int,
    ray_near:float,
    ray_far:float,
    dataset:Dataset, 
    transform_matrix:jnp.ndarray, 
    state:TrainState,
    file_name:Optional[str]='rendered_image'
):    
    get_ray_vmap = jax.vmap(
        jax.vmap(get_ray, in_axes=(None, 0, None, None, None, None, None)), 
        in_axes=(0, None, None, None, None, None, None)
    )
    cpus = jax.devices("cpu")
    t_vals = jnp.linspace(0., 1., num_ray_samples)
    z_vals = ray_near * (1. - t_vals) + ray_far * t_vals
    cpu_z_vals = jax.device_put(z_vals, cpus[0])
    num_patches_x = dataset.w // patch_size_x
    num_patches_y = dataset.h // patch_size_y
    patch_area = patch_size_x * patch_size_y
    rendered_image = np.ones((dataset.w, dataset.h, 3))
    background_colors = jnp.ones((patch_size_x, patch_size_y, 3))

    @jax.jit
    def render_ray(origin, direction, z_vals):
        repeated_origin = jnp.repeat(
            jnp.expand_dims(origin, axis=0), z_vals.shape[0], axis=0
        )
        repeated_direction = jnp.repeat(
            jnp.expand_dims(direction, axis=0), z_vals.shape[0], axis=0
        )
        expanded_z_vals = jnp.expand_dims(z_vals, axis=-1)
        ray_samples = repeated_origin + repeated_direction * expanded_z_vals
        normalized_direction = direction / jnp.linalg.norm(direction, axis=-1)

        def get_output(ray_samples, normalized_directions):
            return state.apply_fn({'params': state.params}, (ray_samples, normalized_directions))
        get_output_sample_vmap = jax.vmap(get_output, in_axes=(0, None))
        densities, colors = get_output_sample_vmap(ray_samples, normalized_direction)
        rendered_pixel = render_pixel(densities, colors, z_vals, direction)
        return rendered_pixel
    render_ray_vmap = jax.vmap(jax.vmap(render_ray, in_axes=0), in_axes=0)
    
    for x in range(num_patches_x):
        patch_start_x = patch_size_x * x
        patch_end_x = patch_start_x + patch_size_x
        x_coordinates = jnp.arange(patch_start_x, patch_end_x)
        for y in range(num_patches_y):
            patch_start_y = patch_size_y * y
            patch_end_y = patch_start_y + patch_size_y
            y_coordinates = jnp.arange(patch_start_y, patch_end_y)
            
            ray_origins, ray_directions = get_ray_vmap(
                x_coordinates, 
                y_coordinates, 
                transform_matrix,
                dataset.cx,
                dataset.cy,
                dataset.fl_x,
                dataset.fl_y
            )
            new_shape = (patch_area, 3)
            cpu_ray_origins = jax.device_put(jnp.reshape(ray_origins, new_shape), cpus[0])
            cpu_ray_directions = jax.device_put(jnp.reshape(ray_directions, new_shape), cpus[0])
            new_z_vals = jnp.array(batch_trace_rays(
                cpu_ray_origins, cpu_ray_directions, cpu_z_vals, patch_area, num_ray_samples
            ))
            new_z_vals = jnp.reshape(new_z_vals, (patch_size_x, patch_size_y, num_ray_samples))
            rendered_colors, rendered_alphas = render_ray_vmap(
                ray_origins, ray_directions, new_z_vals
            )
            rendered_patch = alpha_composite(rendered_colors, background_colors, rendered_alphas)
            rendered_image[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = rendered_patch

    rendered_image = np.nan_to_num(rendered_image)
    rendered_image = np.clip(rendered_image, 0, 1)
    plt.imsave(os.path.join('data/', file_name + '.png'), rendered_image.transpose((1, 0, 2)))

def generate_density_grid(
    num_points:int, 
    patch_size_x:int, 
    patch_size_y:int,
    state:TrainState,
    file_name:Optional[str]='density_grid'
):
    @jax.jit
    def get_density(x, y):
        def get_output(rays, directions):
            return state.apply_fn({'params': state.params}, (rays, directions))
        get_output_sample_vmap = jax.vmap(get_output, in_axes=(0, None))

        rays = jnp.repeat(jnp.array([[x, y]]), num_points, axis=0)
        rays_z = jnp.expand_dims(jnp.linspace(0, 1, num_points), axis=-1)
        rays = jnp.concatenate([rays, rays_z], axis=-1)
        direction = jnp.array([0, 0, 1])
        density, _ = get_output_sample_vmap(rays, direction)
        return density
    
    num_patches_x = num_points // patch_size_x
    num_patches_y = num_points // patch_size_y
    density_grid = np.zeros((num_points, num_points, num_points, 1))
    get_density_vmap = jax.vmap(jax.vmap(get_density, in_axes=(None, 0)), in_axes=(0, None))

    for x in range(num_patches_x):
        patch_start_x = patch_size_x * x
        patch_end_x = patch_start_x + patch_size_x
        x_coordinates = jnp.arange(patch_start_x, patch_end_x) / num_points
        for y in range(num_patches_y):
            patch_start_y = patch_size_y * y
            patch_end_y = patch_start_y + patch_size_y
            y_coordinates = jnp.arange(patch_start_y, patch_end_y) / num_points
            density_patch = get_density_vmap(x_coordinates, y_coordinates)
            density_grid[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = density_patch

    print('Density grid shape:', density_grid.shape)
    np.save('data/density_grid.npy', density_grid)

def turntable_render(
    num_frames:int, 
    num_ray_samples:int,
    patch_size_x:int,
    patch_size_y:int,
    camera_distance:float, 
    ray_near:float, 
    ray_far:float, 
    state:TrainState, 
    dataset:Dataset,
    file_name:str='turntable_render'
):
    xy_start_position = jnp.array([0.0, -1.0])
    xy_start_position_angle_2d = 0
    z_start_rotation_angle_3d = 0
    angle_delta = 2 * jnp.pi / num_frames

    x_rotation_angle_3d = jnp.pi / 2
    x_rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(x_rotation_angle_3d), -jnp.sin(x_rotation_angle_3d)],
        [0, jnp.sin(x_rotation_angle_3d), jnp.cos(x_rotation_angle_3d)],
    ])

    for i in range(num_frames):
        xy_position_angle_2d = xy_start_position_angle_2d + i * angle_delta
        z_rotation_angle_3d = z_start_rotation_angle_3d + i * angle_delta

        xy_rotation_matrix_2d = jnp.array([
            [jnp.cos(xy_position_angle_2d), -jnp.sin(xy_position_angle_2d)], 
            [jnp.sin(xy_position_angle_2d), jnp.cos(xy_position_angle_2d)]
        ])
        current_xy_position = xy_rotation_matrix_2d @ xy_start_position
    
        z_rotation_matrix = jnp.array([
            [jnp.cos(z_rotation_angle_3d), -jnp.sin(z_rotation_angle_3d), 0],
            [jnp.sin(z_rotation_angle_3d), jnp.cos(z_rotation_angle_3d), 0],
            [0, 0, 1],
        ])

        rotation_matrix = z_rotation_matrix @ x_rotation_matrix
        translation_matrix = jnp.array([
            [current_xy_position[0]],
            [current_xy_position[1]],
            [0],
        ])
        transform_matrix = jnp.concatenate([rotation_matrix, translation_matrix], axis=-1)
        transform_matrix = process_3x4_transform_matrix(transform_matrix, camera_distance)

        render_scene(
            num_ray_samples=num_ray_samples, 
            patch_size_x=patch_size_x, 
            patch_size_y=patch_size_y, 
            ray_near=ray_near, 
            ray_far=ray_far, 
            dataset=dataset, 
            transform_matrix=transform_matrix, 
            state=state,
            file_name=file_name + f'_frame_{i}'
        )

def process_3x4_transform_matrix(original:jnp.ndarray, scale:float):    
    new = jnp.array([
        [original[1, 0], -original[1, 1], -original[1, 2], original[1, 3] * scale + 0.5],
        [original[2, 0], -original[2, 1], -original[2, 2], original[2, 3] * scale + 0.5],
        [original[0, 0], -original[0, 1], -original[0, 2], original[0, 3] * scale + 0.5],
    ])
    return new

def load_dataset(path:str, downscale_factor:int, translation_scale:float):
    with open(os.path.join(path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        image = Image.open(os.path.join(path, frame['file_path']))
        image = image.resize(
            (image.width // downscale_factor, image.height // downscale_factor),
            resample=Image.NEAREST
        )
        images.append(jnp.array(image))

    transform_matrices = jnp.array(transform_matrices)[:, :3, :]
    process_transform_matrices_vmap = jax.vmap(process_3x4_transform_matrix, in_axes=(0, None))
    transform_matrices = process_transform_matrices_vmap(transform_matrices, translation_scale)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    dataset = Dataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_y'],
        fl_x=1,
        fl_y=1,
        k1=transforms['k1'],
        k2=transforms['k2'],
        p1=transforms['p1'],
        p2=transforms['p2'],
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        aabb_scale=transforms['aabb_scale'],
        transform_matrices=transform_matrices,
        images=images
    )
    dataset.fl_x = dataset.cx / jnp.tan(dataset.horizontal_fov / 2)
    dataset.fl_y = dataset.cy / jnp.tan(dataset.vertical_fov / 2)
    return dataset

def load_lego_dataset(path:str, downscale_factor:int, translation_scale:float):
    with open(os.path.join(path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        current_image_path = path + frame['file_path'][1:] + '.png'
        image = Image.open(current_image_path)
        image = image.resize(
            (image.width // downscale_factor, image.height // downscale_factor),
            resample=Image.NEAREST
        )
        images.append(jnp.array(image))

    transform_matrices = jnp.array(transform_matrices)[:, :3, :]
    process_transform_matrices_vmap = jax.vmap(process_3x4_transform_matrix, in_axes=(0, None))
    transform_matrices = process_transform_matrices_vmap(transform_matrices, translation_scale)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    dataset = Dataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_x'],
        fl_x=1,
        fl_y=1,
        k1=0,
        k2=0,
        p1=0,
        p2=0,
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        aabb_scale=0,
        transform_matrices=transform_matrices,
        images=images
    )
    dataset.fl_x = dataset.cx / jnp.tan(dataset.horizontal_fov / 2)
    dataset.fl_y = dataset.cy / jnp.tan(dataset.vertical_fov / 2)
    return dataset

if __name__ == '__main__':
    print('GPU:', jax.devices('gpu'))

    #dataset = load_lego_dataset('data/lego', 0.33)
    dataset = load_lego_dataset('data/lego', 8, 0.33)
    #dataset = load_dataset('data/generation_0', 0.1)
    print(dataset.horizontal_fov)
    print(dataset.vertical_fov)
    print(dataset.fl_x)
    print(dataset.fl_y)
    print(dataset.k1)
    print(dataset.k2)
    print(dataset.p1)
    print(dataset.p2)
    print(dataset.cx)
    print(dataset.cy)
    print(dataset.w)
    print(dataset.h)
    print(dataset.aabb_scale)
    print('Images shape:', dataset.images.shape)

    model = InstantNerf(
        number_of_grid_levels=16,
        max_hash_table_entries=2**20,
        hash_table_feature_dim=2,
        coarsest_resolution=16,
        finest_resolution=2**19,
        density_mlp_width=64,
        color_mlp_width=64,
        high_dynamic_range=False,
        exponential_density_activation=False
    )
    rng = jax.random.PRNGKey(1)
    state = create_train_state(model, rng, 1e-2, 10**-15)

    ray_near = 0.1
    ray_far = 3.0
    assert ray_near < ray_far, 'Ray near must be less than ray far.'

    state = train_loop(
        batch_size=4096,
        num_ray_samples=128,
        ray_near=ray_near,
        ray_far=ray_far,
        training_steps=10000, 
        state=state, 
        dataset=dataset
    )
    #turntable_render(
    #    10, 128, 32, 32, 0.5, ray_near, ray_far, state, dataset, 'instant_turntable_render'
    #)
    #print('Finished turntable')
    #generate_density_grid(128, 32, 32, state)
    #print('Finished density')
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[9], 
        state=state,
        file_name='instant_rendered_image_0'
    )
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[14], 
        state=state,
        file_name='instant_rendered_image_1'
    )
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[7], 
        state=state,
        file_name='instant_rendered_image_2'
    )
    