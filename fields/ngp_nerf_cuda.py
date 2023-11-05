import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from volrendjax import integrate_rays, march_rays
from volrendjax import morton3d_invert, packbits
from volrendjax import make_near_far_from_bound
from dataclasses import dataclass
from functools import partial
import optax
import json
from PIL import Image
from fields import ngp_nerf, Dataset, trunc_exp
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Callable
import time
from jaxtcnn import hashgrid_encode, HashGridMetadata
import math

@dataclass
class OccupancyGrid:
    resolution: int
    num_entries: int
    update_interval: int
    warmup_steps: int
    densities: jax.Array # Full precision density values.
    mask: jax.Array # Non-compact boolean representation of occupancy.
    bitfield: jax.Array # Compact occupancy bitfield.

def create_occupancy_grid(resolution:int, update_interval:int, warmup_steps:int):
    num_entries = resolution**3
    # Each bit is an occupancy value, and uint8 is 8 bytes, so divide num_entries by 8.
    # This gives one entry per bit.
    bitfield = 255 * jnp.ones(shape=(num_entries // 8,), dtype=jnp.uint8)
    densities = jnp.zeros(shape=(num_entries,), dtype=jnp.float32)
    mask = jnp.zeros(shape=(num_entries,), dtype=jnp.bool_)
    return OccupancyGrid(
        resolution, num_entries, update_interval, 
        warmup_steps, densities, mask, bitfield
    )

def update_occupancy_grid_density(
    KEY, batch_size:int, densities:jax.Array, occupancy_mask:jax.Array, grid_resolution: int, 
    num_grid_entries:int, scene_bound:float, state:TrainState, warmup:bool
):
    decayed_densities = densities * .95
    all_indices = jnp.arange(num_grid_entries)
    if warmup:
        updated_indices = all_indices
    else:
        quarter_num_grid_entries = num_grid_entries // 4 # Corresponds to M/2 in the paper.
        KEY, first_half_key, second_half_key = jax.random.split(KEY, 3)
        # Uniformly sample M/2 grid cells.
        uniform_index_samples = jax.random.choice(
            key=first_half_key, 
            a=all_indices, 
            shape=(quarter_num_grid_entries,), 
            replace=True, # Allow duplicate choices. 
        )
        # Uniformly sample M/2 occupied grid cells.
        uniform_occupied_index_samples = jax.random.choice(
            key=second_half_key, 
            a=all_indices, 
            shape=(quarter_num_grid_entries,), 
            replace=True, # Allow duplicate choices. 
            p=occupancy_mask.astype(jnp.float32) # Only sample from occupied cells.
        )
        # Total of M samples, where M is num_grid_entries / 2.
        updated_indices = jnp.concatenate([
            uniform_index_samples, uniform_occupied_index_samples
        ])

    updated_indices = updated_indices.astype(jnp.uint32)
    coordinates = morton3d_invert(updated_indices).astype(jnp.float32)
    # Transform coordinates to [-1, 1].
    coordinates = coordinates / (grid_resolution - 1) * 2 - 1
    half_cell_width = scene_bound / grid_resolution
    # Transform coordinates to [-scene_bound + half_cell_width, scene_bound - half_cell_width].
    coordinates = coordinates * (scene_bound - half_cell_width)
    # Get random points inside grid cells.
    KEY, key = jax.random.split(KEY, 2)
    coordinates = coordinates + jax.random.uniform(
        key,
        coordinates.shape,
        coordinates.dtype,
        minval=-half_cell_width,
        maxval=half_cell_width,
    )
    
    # [num_updated_entries,]
    updated_densities = jnp.ravel(
        state.apply_fn({'params': state.params}, (coordinates, None))
    )
    updated_densities = jnp.maximum(decayed_densities[updated_indices], updated_densities)
    # [num_grid_entries,]
    updated_densities = decayed_densities.at[updated_indices].set(updated_densities)
    return updated_densities

@jax.jit
def threshold_occupancy_grid(diagonal_n_steps:int, scene_bound:float, densities:jax.Array):
    def density_threshold_from_min_step_size(diagonal_n_steps, scene_bound) -> float:
        return .01 * diagonal_n_steps / (2 * jnp.minimum(scene_bound, 1) * 3**.5)
    
    density_threshold = jnp.minimum(
        density_threshold_from_min_step_size(diagonal_n_steps, scene_bound),
        jnp.mean(densities)
    )

    occupancy_mask, occupancy_bitfield = packbits(
        density_threshold=density_threshold,
        density_grid=densities
    )
    return occupancy_mask, occupancy_bitfield

def create_train_state(
    model:nn.Module, 
    rng,
    learning_rate:float, 
    epsilon:float, 
    weight_decay_coefficient:float
):
    x = (jnp.ones([1, 3]) / 3, jnp.ones([1, 3]) / 3)
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
    # The alt scale is just to debug the effects of the cameras being closer to or further away
    # from the origin.
    alt_scale = 2
    new = jnp.array([
        [original[1, 0], -original[1, 1], -original[1, 2], original[1, 3] * scale * alt_scale],
        [original[2, 0], -original[2, 1], -original[2, 2], original[2, 3] * scale * alt_scale],
        [original[0, 0], -original[0, 1], -original[0, 2], original[0, 3] * scale * alt_scale],
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

class MultiResolutionHashEncoding(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.hash_offset = jnp.concatenate([
            self.hash_offset, jnp.array([self.hash_offset[-1] + self.table_size])
        ], axis=0)
        self.spatial_dim = 3
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1.0
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        self.scalings = jnp.reshape(self.scalings, (self.scalings.shape[0], 1))
        absolute_table_size = self.table_size * self.num_levels
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (absolute_table_size, self.feature_dim,)
        )

    def __call__(self, x):
        _growth_factor = math.exp(
            (math.log(self.max_resolution) - math.log(self.min_resolution)) 
            / (self.num_levels - 1)
        )
        encoded_position = hashgrid_encode(
            desc=HashGridMetadata(
                L=int(self.num_levels),
                F=int(self.feature_dim),
                N_min=int(self.min_resolution),
                per_level_scale=_growth_factor
            ),
            offset_table_data=jnp.asarray(self.hash_offset, jnp.uint32),
            coords_rm=x.T,
            params=self.hash_table
        )
        return encoded_position.T

class NGPNerf(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.
    density_mlp_width: int
    color_mlp_width: int
    high_dynamic_range: bool
    exponential_density_activation: bool
    scene_bound: float

    @nn.compact
    def __call__(self, x):
        position, direction = x
        position = (position + self.scene_bound) / (2.0 * self.scene_bound)
        encoded_position = MultiResolutionHashEncoding(
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim
        )(position)
        x = encoded_position

        x = nn.Dense(self.density_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(16, kernel_init=nn.initializers.normal())(x)
        if self.exponential_density_activation:
            density = trunc_exp(x[:, 0:1])
        else:
            density = nn.activation.relu(x[:, 0:1])
        if direction is None:
            return density
        density_feature = x

        encoded_direction = jax.vmap(ngp_nerf.fourth_order_sh_encoding, in_axes=0)(direction)
        x = jnp.concatenate([density_feature, encoded_direction], axis=-1)
        x = nn.Dense(self.color_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(self.color_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(3)(x)

        if self.high_dynamic_range:
            color = jnp.exp(x)
        else:
            color = nn.activation.sigmoid(x)
        drgbs = jnp.concatenate([density, color], axis=-1)
        return drgbs

def update_occupancy_grid(
    batch_size:int, diagonal_n_steps:int, scene_bound:float, step:int,
    state:TrainState, occupancy_grid:OccupancyGrid
):
    warmup = step < occupancy_grid.warmup_steps
    occupancy_grid.densities = jax.lax.stop_gradient(
        update_occupancy_grid_density(
            KEY=jax.random.PRNGKey(step),
            batch_size=batch_size,
            densities=occupancy_grid.densities,
            occupancy_mask=occupancy_grid.mask,
            grid_resolution=occupancy_grid.resolution,
            num_grid_entries=occupancy_grid.num_entries,
            scene_bound=scene_bound,
            state=state,
            warmup=warmup
        )
    )
    occupancy_grid.mask, occupancy_grid.bitfield = jax.lax.stop_gradient(
        threshold_occupancy_grid(
            diagonal_n_steps=diagonal_n_steps,
            scene_bound=scene_bound,
            densities=occupancy_grid.densities
        )
    )
    return occupancy_grid

def train_loop(
    batch_size:int, 
    train_steps:int, 
    dataset:Dataset, 
    scene_bound:float, 
    diagonal_n_steps:int, 
    stepsize_portion:float,
    occupancy_grid:OccupancyGrid,
    state:TrainState,
    return_final_loss:bool=False
):
    num_images = dataset.images.shape[0]
    for step in range(train_steps):
        loss, state, n_rays = train_step(
            KEY=jax.random.PRNGKey(step),
            batch_size=batch_size,
            image_width=dataset.w,
            image_height=dataset.h,
            num_images=num_images,
            images=dataset.images,
            transform_matrices=dataset.transform_matrices,
            state=state,
            occupancy_bitfield=occupancy_grid.bitfield,
            grid_resolution=occupancy_grid.resolution,
            principal_point_x=dataset.cx,
            principal_point_y=dataset.cy,
            focal_length_x=dataset.fl_x,
            focal_length_y=dataset.fl_y,
            scene_bound=scene_bound,
            diagonal_n_steps=diagonal_n_steps,
            stepsize_portion=stepsize_portion
        )
        #print('Step', step, 'Loss', loss, 'N Rays', n_rays)
        if step % occupancy_grid.update_interval == 0 and step > 0:
            #print('Updating occupancy grid...')
            occupancy_grid = update_occupancy_grid(
                batch_size=batch_size,
                diagonal_n_steps=diagonal_n_steps,
                scene_bound=scene_bound,
                step=step,
                state=state,
                occupancy_grid=occupancy_grid
            )
    if return_final_loss:
        return state, occupancy_grid, loss
    else:
        return state, occupancy_grid

def sample_pixels(key, num_samples:int, image_width:int, image_height:int, num_images:int):
    width_rng, height_rng, image_rng = jax.random.split(key, num=3) 
    width_indices = jax.random.randint(width_rng, (num_samples,), 0, image_width)
    height_indices = jax.random.randint(height_rng, (num_samples,), 0, image_height)
    image_indices = jax.random.randint(image_rng, (num_samples,), 0, num_images)
    return image_indices, width_indices, height_indices 

@jax.jit
def get_ray(uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    direction = direction / jnp.linalg.norm(direction)
    origin = transform_matrix[:3, -1]
    return origin, direction

@partial(jax.jit, static_argnames=(
    'batch_size', 'image_width', 'image_height', 'num_images', 'grid_resolution',
    'diagonal_n_steps', 'scene_bound', 'stepsize_portion'
))
def train_step(
    KEY, batch_size:int, image_width:int, image_height:int, num_images:int, images:jax.Array,
    transform_matrices:jax.Array, state:TrainState, occupancy_bitfield:jax.Array, 
    grid_resolution:int, principal_point_x:float, principal_point_y:float, 
    focal_length_x:float, focal_length_y:float, scene_bound:float, diagonal_n_steps:int,
    stepsize_portion:float
):
    pixel_sample_key, random_bg_key, noise_key = jax.random.split(KEY, 3)
    image_indices, width_indices, height_indices = sample_pixels(
        key=pixel_sample_key, num_samples=batch_size, image_width=image_width,
        image_height=image_height, num_images=num_images
    )

    get_rays = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    ray_origins, ray_directions = get_rays(
        width_indices, height_indices, transform_matrices[image_indices], 
        principal_point_x, principal_point_y, focal_length_x, focal_length_y
    )

    t_starts, t_ends = make_near_far_from_bound(scene_bound, ray_origins, ray_directions)
    noises = jax.random.uniform(
        noise_key, (batch_size,), dtype=t_starts.dtype, minval=0.0, maxval=1.0
    )

    ray_march_result = march_rays(
        total_samples=batch_size,
        diagonal_n_steps=diagonal_n_steps,
        K=1,
        G=grid_resolution,
        bound=scene_bound,
        stepsize_portion=stepsize_portion,
        rays_o=ray_origins,
        rays_d=ray_directions,
        t_starts=jnp.ravel(t_starts),
        t_ends=jnp.ravel(t_ends),
        noises=noises,
        occupancy_bitfield=occupancy_bitfield
    )

    _, ray_is_valid, rays_n_samples, rays_sample_start_idx, \
    _, positions, directions, dss, z_vals = ray_march_result
    num_valid_rays = jnp.sum(ray_is_valid)

    def compute_sample(params, ray_sample, direction):
        return state.apply_fn({'params': params}, (ray_sample, direction))
    #compute_batch = jax.vmap(compute_sample, in_axes=(None, 0, 0))
    compute_batch = compute_sample

    def loss_fn(params):
        drgbs = compute_batch(params, positions, directions)
        background_colors = jax.random.uniform(random_bg_key, (batch_size, 3))
        integration_result = integrate_rays(
            near_distance=0.3,
            rays_sample_startidx=rays_sample_start_idx,
            rays_n_samples=rays_n_samples,
            bgs=background_colors,
            dss=dss,
            z_vals=z_vals,
            drgbs=drgbs,
        )
        _, final_rgbds, _ = integration_result
        pred_rgbs, _ = jnp.array_split(final_rgbds, [3], axis=-1)
        target_pixels = images[image_indices, height_indices, width_indices]
        target_rgbs = target_pixels[:, :3]
        target_alphas = target_pixels[:, 3:]
        target_rgbs = target_rgbs * target_alphas + background_colors * (1.0 - target_alphas)
        loss = jnp.sum(jnp.where(
            ray_is_valid, 
            jnp.mean(optax.huber_loss(pred_rgbs, target_rgbs, delta=0.1), axis=-1),
            0.0,
        )) / num_valid_rays
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state, num_valid_rays


@partial(jax.jit, static_argnames=(
    'principal_point_x',
    'principal_point_y',
    'focal_length_x',
    'focal_length_y',
    'scene_bound',
    'diagonal_n_steps',
    'grid_cascades',
    'grid_resolution',
    'stepsize_portion',
    'total_ray_samples',
    'max_num_rays'
))
def render_rays_inference(
    width_indices:jax.Array, height_indices:jax.Array, transform_matrix:jax.Array, 
    principal_point_x:int, principal_point_y:int, focal_length_x:float, focal_length_y:float,
    scene_bound:float, diagonal_n_steps:int, grid_cascades:int, grid_resolution:int, 
    stepsize_portion:float, total_ray_samples:int, occupancy_bitfield:jax.Array, 
    state:TrainState, max_num_rays:int
):
    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, None, None, None, None, None))
    ray_origins, ray_directions = get_ray_vmap(
        width_indices, height_indices, transform_matrix, 
        principal_point_x, principal_point_y, focal_length_x, focal_length_y
    )
    t_starts, t_ends = make_near_far_from_bound(
        bound=scene_bound, o=ray_origins, d=ray_directions
    )
    noises = jnp.zeros((ray_origins.shape[0],))

    (measured_batch_size_before_compaction, ray_is_valid, rays_n_samples,
    rays_sample_startidx, ray_idcs, xyzs, dirs, dss, z_vals) = march_rays(
        total_samples=total_ray_samples, 
        diagonal_n_steps=diagonal_n_steps,
        K=grid_cascades,
        G=grid_resolution,
        bound=scene_bound,
        stepsize_portion=stepsize_portion,
        rays_o=ray_origins,
        rays_d=ray_directions,
        t_starts=jnp.ravel(t_starts),
        t_ends=jnp.ravel(t_ends),
        noises=noises,
        occupancy_bitfield=occupancy_bitfield
    )

    drgbs = state.apply_fn({'params': state.params}, (xyzs, dirs))    
    background_colors = jnp.ones((max_num_rays, 3))
    effective_samples, final_rgbds, final_opacities = integrate_rays(
        near_distance=0.1,
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=background_colors,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )
    pred_rgbs, pred_depths = jnp.array_split(final_rgbds, [3], axis=-1)
    return pred_rgbs, pred_depths

def render_scene(
    patch_size_x:int,
    patch_size_y:int,
    dataset:Dataset, 
    scene_bound:float,
    diagonal_n_steps:int,
    grid_cascades:int,
    grid_resolution:int,
    stepsize_portion:float,
    occupancy_bitfield:jax.Array,
    transform_matrix:jnp.ndarray, 
    batch_size:int,
    state:TrainState,
    file_name:Optional[str]='rendered_image'
):    
    num_patches_x = dataset.w // patch_size_x
    num_patches_y = dataset.h // patch_size_y
    patch_area = patch_size_x * patch_size_y
    image = np.ones((dataset.w, dataset.h, 3), dtype=np.float32)
    depth_map = np.ones((dataset.w, dataset.h, 1), dtype=np.float32)

    for x in range(num_patches_x):
        patch_start_x = patch_size_x * x
        patch_end_x = patch_start_x + patch_size_x
        x_coordinates = jnp.arange(patch_start_x, patch_end_x)
        for y in range(num_patches_y):
            patch_start_y = patch_size_y * y
            patch_end_y = patch_start_y + patch_size_y
            y_coordinates = jnp.arange(patch_start_y, patch_end_y)
            
            x_grid_coordinates, y_grid_coordinates = jnp.meshgrid(x_coordinates, y_coordinates)
            x_grid_coordinates = jnp.ravel(x_grid_coordinates)
            y_grid_coordinates = jnp.ravel(y_grid_coordinates)

            rendered_colors, rendered_depths = render_rays_inference(
                width_indices=x_grid_coordinates, 
                height_indices=y_grid_coordinates, 
                transform_matrix=transform_matrix, 
                principal_point_x=dataset.cx,
                principal_point_y=dataset.cy, 
                focal_length_x=dataset.fl_x, 
                focal_length_y=dataset.fl_y, 
                scene_bound=scene_bound, 
                diagonal_n_steps=diagonal_n_steps, 
                grid_cascades=grid_cascades, 
                grid_resolution=grid_resolution, 
                stepsize_portion=stepsize_portion, 
                total_ray_samples=batch_size, 
                occupancy_bitfield=occupancy_bitfield, 
                state=state, 
                max_num_rays=patch_area
            )

            image_patch_shape = (patch_size_x, patch_size_y, 3)
            image_patch = np.reshape(rendered_colors, image_patch_shape, order='F')
            image[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = image_patch

            depth_patch_shape = (patch_size_x, patch_size_y, 1)
            depth_patch = np.reshape(rendered_depths, depth_patch_shape, order='F')
            depth_map[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = depth_patch

    image = np.nan_to_num(image)
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 0, 2))
    plt.imsave(os.path.join('data/', file_name + '.png'), image)
    depth_map = np.nan_to_num(depth_map)
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = np.transpose(depth_map, (1, 0, 2))
    depth_map = np.squeeze(depth_map, axis=-1)
    plt.imsave(os.path.join('data/', file_name + '_depth.png'), depth_map, cmap='gray')

def turntable_render(
    num_frames:int, 
    camera_distance:float, 
    render_fn:Callable,
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
        render_fn(transform_matrix=transform_matrix, file_name=file_name + f'_frame_{i}')

def benchmark_training(train_loop_with_args:Callable):
    start_time = time.time()
    state, occupancy_grid = train_loop_with_args()
    end_time = time.time()
    print('Training time:', end_time - start_time)

def main():
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = True

    learning_rate = 1e-2
    epsilon = 1e-15
    weight_decay_coefficient = 1e-6
    batch_size = 256 * 1024
    scene_bound = 1.0
    grid_resolution = 128
    grid_update_interval = 16
    grid_warmup_steps = 256
    diagonal_n_steps = 1024
    train_steps = 1000
    stepsize_portion = 1.0 / 256.0

    model = NGPNerf(
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
    KEY, state_init_key = jax.random.split(KEY, num=2)
    state = create_train_state(
        model=model, 
        rng=state_init_key, 
        learning_rate=learning_rate,
        epsilon=epsilon,
        weight_decay_coefficient=weight_decay_coefficient
    )
    occupancy_grid = create_occupancy_grid(
        resolution=grid_resolution, 
        update_interval=grid_update_interval, 
        warmup_steps=grid_warmup_steps
    )
    dataset = load_dataset('data/lego', 1)
    train_loop_with_args = partial(
        train_loop,
        batch_size=batch_size,
        train_steps=train_steps,
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        stepsize_portion=stepsize_portion,
        occupancy_grid=occupancy_grid,
        state=state
    )
    #benchmark_training(train_loop_with_args)
    #exit(0)
    
    state, occupancy_grid = train_loop_with_args()

    render_fn = partial(
        render_scene,
        # Patch size has to be small otherwise not all rays will produce samples and the
        # resulting image will have artifacts. This can be fixed by switching to the 
        # inference version of the ray marching and ray integration functions.
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
    render_fn(
        transform_matrix=dataset.transform_matrices[3],
        file_name='ngp_nerf_cuda_rendered_image'
    )
    #turntable_render(
    #    num_frames=60*3,
    #    camera_distance=1,
    #    render_fn=render_fn,
    #    file_name='ngp_nerf_cuda_turntable_render'
    #)
    jnp.save('data/occupancy_grid_density.npy', occupancy_grid.mask.astype(jnp.float32))
    occupancy_grid_coordinates = morton3d_invert(
        jnp.arange(occupancy_grid.mask.shape[0], dtype=jnp.uint32)
    )
    occupancy_grid_coordinates = occupancy_grid_coordinates / (grid_resolution - 1) * 2 - 1
    jnp.save('data/occupancy_grid_coordinates.npy', occupancy_grid_coordinates)

if __name__ == '__main__':
    main()