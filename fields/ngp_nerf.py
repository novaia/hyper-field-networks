import os
import time
import json
from functools import partial
from typing import Optional, Callable, Tuple, Union
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from fields.common.nn import \
    TcnnMultiResolutionHashEncoding, FeedForward, fourth_order_sh_encoding, trunc_exp
from fields.common.dataset import NerfDataset, load_nerf_dataset
from fields.common.matrices import get_z_axis_camera_orbit_matrix
from volrendjax import \
    integrate_rays, march_rays, morton3d_invert, packbits, make_near_far_from_bound
import matplotlib.pyplot as plt

@dataclass
class OccupancyGrid:
    resolution: int
    num_entries: int
    update_interval: int
    warmup_steps: int
    densities: jax.Array # Full precision density values.
    mask: jax.Array # Non-compact boolean representation of occupancy.
    bitfield: jax.Array # Compact occupancy bitfield.

def create_occupancy_grid(
    resolution:int, update_interval:int, warmup_steps:int
) -> OccupancyGrid:
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

def update_occupancy_grid(
    batch_size:int, diagonal_n_steps:int, scene_bound:float, step:int,
    state:TrainState, occupancy_grid:OccupancyGrid
) -> OccupancyGrid:
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

def update_occupancy_grid_density(
    KEY, densities:jax.Array, occupancy_mask:jax.Array, grid_resolution: int, 
    num_grid_entries:int, scene_bound:float, state:TrainState, warmup:bool
) -> jax.Array:
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
        key, coordinates.shape, coordinates.dtype, 
        minval=-half_cell_width, maxval=half_cell_width,
    )
    
    # [num_updated_entries,]
    updated_densities = jnp.ravel(state.apply_fn({'params': state.params}, (coordinates,None)))
    updated_densities = jnp.maximum(decayed_densities[updated_indices], updated_densities)
    # [num_grid_entries,]
    updated_densities = decayed_densities.at[updated_indices].set(updated_densities)
    return updated_densities

@jax.jit
def threshold_occupancy_grid(
    diagonal_n_steps:int, scene_bound:float, densities:jax.Array
) -> Tuple[jax.Array, jax.Array]:
    def density_threshold_from_min_step_size(diagonal_n_steps, scene_bound) -> float:
        return .01 * diagonal_n_steps / (2 * jnp.minimum(scene_bound, 1) * 3**.5)
    
    density_threshold = jnp.minimum(
        density_threshold_from_min_step_size(diagonal_n_steps, scene_bound),
        jnp.mean(densities)
    )

    occupancy_mask, occupancy_bitfield = packbits(density_threshold, densities)
    return occupancy_mask, occupancy_bitfield

def create_train_state(
    model:nn.Module, rng, learning_rate:float, epsilon:float, weight_decay_coefficient:float
) -> TrainState:
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
        encoded_position = TcnnMultiResolutionHashEncoding(
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim
        )(position)
        x = encoded_position

        x = FeedForward(
            num_layers=1, hidden_dim=self.density_mlp_width, output_dim=16, activation=nn.relu
        )(x)
        density = x[:, 0:1]
        if self.exponential_density_activation: density = trunc_exp(density)
        else: density = nn.relu(density)
        if direction is None: return density
        density_feature = x

        encoded_direction = jax.vmap(fourth_order_sh_encoding, in_axes=0)(direction)
        x = jnp.concatenate([density_feature, encoded_direction], axis=-1)
        x = FeedForward(
            num_layers=2, hidden_dim=self.color_mlp_width, output_dim=3, activation=nn.relu
        )(x)

        if self.high_dynamic_range: color = jnp.exp(x)
        else: color = nn.activation.sigmoid(x)
        drgbs = jnp.concatenate([density, color], axis=-1)
        return drgbs

def train_loop(
    batch_size:int, train_steps:int, dataset:NerfDataset, scene_bound:float, 
    diagonal_n_steps:int, stepsize_portion:float, occupancy_grid:OccupancyGrid,
    state:TrainState, return_final_loss:bool=False
) -> Union[Tuple[TrainState, OccupancyGrid], Tuple[TrainState, OccupancyGrid, float]]:
    num_images = dataset.images.shape[0]
    for step in range(train_steps):
        loss, state = train_step(
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
        if step % occupancy_grid.update_interval == 0 and step > 0:
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
) -> Tuple[float, TrainState]:
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
    noises = jax.random.uniform(noise_key, (batch_size,), t_starts.dtype, minval=0., maxval=1.)

    _, ray_is_valid, rays_n_samples, rays_sample_start_idx, \
    _, positions, directions, dss, z_vals = march_rays(
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
    num_valid_rays = jnp.sum(ray_is_valid)

    def loss_fn(params):
        drgbs = state.apply_fn({'params': params}, (positions, directions))
        background_colors = jax.random.uniform(random_bg_key, (batch_size, 3))
        _, final_rgbds, _ = integrate_rays(
            near_distance=0.3,
            rays_sample_startidx=rays_sample_start_idx,
            rays_n_samples=rays_n_samples,
            bgs=background_colors,
            dss=dss,
            z_vals=z_vals,
            drgbs=drgbs,
        )
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
    return loss, state

@partial(jax.jit, static_argnames=(
    'principal_point_x', 'principal_point_y', 'focal_length_x', 'focal_length_y', 
    'scene_bound', 'diagonal_n_steps', 'grid_cascades', 'grid_resolution', 'stepsize_portion',
    'total_ray_samples', 'max_num_rays'
))
def render_rays_inference(
    width_indices:jax.Array, height_indices:jax.Array, transform_matrix:jax.Array, 
    principal_point_x:int, principal_point_y:int, focal_length_x:float, focal_length_y:float,
    scene_bound:float, diagonal_n_steps:int, grid_cascades:int, grid_resolution:int, 
    stepsize_portion:float, total_ray_samples:int, occupancy_bitfield:jax.Array, 
    state:TrainState, max_num_rays:int
) -> Tuple[jax.Array, jax.Array]:
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
    patch_size_x:int, patch_size_y:int, dataset:NerfDataset, scene_bound:float,
    diagonal_n_steps:int, grid_cascades:int, grid_resolution:int, stepsize_portion:float,
    occupancy_bitfield:jax.Array, transform_matrix:jnp.ndarray, batch_size:int, 
    state:TrainState, file_name:Optional[str]='rendered_image'
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
    num_frames:int, camera_distance:float, render_fn:Callable, file_name:str='turntable_render'
):
    angle_delta = 2 * jnp.pi / num_frames
    for i in range(num_frames):
        camera_matrix = get_z_axis_camera_orbit_matrix(i * angle_delta, camera_distance)
        render_fn(transform_matrix=camera_matrix, file_name=file_name + f'_frame_{i}')

def main():
    with open('configs/ngp_nerf.json', 'r') as f:
        config = json.load(f)

    model = NGPNerf(
        number_of_grid_levels=config['num_hash_table_levels'],
        max_hash_table_entries=config['max_hash_table_entries'],
        hash_table_feature_dim=config['hash_table_feature_dim'],
        coarsest_resolution=config['coarsest_resolution'],
        finest_resolution=config['finest_resolution'],
        density_mlp_width=config['density_mlp_width'],
        color_mlp_width=config['color_mlp_width'],
        high_dynamic_range=config['high_dynamic_range'],
        exponential_density_activation=config['exponential_density_activation'],
        scene_bound=config['scene_bound']
    )
    KEY = jax.random.PRNGKey(0)
    KEY, state_init_key = jax.random.split(KEY, num=2)
    state = create_train_state(
        model=model, 
        rng=state_init_key, 
        learning_rate=config['learning_rate'],
        epsilon=config['epsilon'],
        weight_decay_coefficient=config['weight_decay_coefficient']
    )
    occupancy_grid = create_occupancy_grid(
        resolution=config['grid_resolution'], 
        update_interval=config['grid_update_interval'], 
        warmup_steps=config['grid_warmup_steps']
    )
    dataset = load_nerf_dataset('data/lego', 1)
    train_loop_with_args = partial(
        train_loop,
        batch_size=config['batch_size'],
        train_steps=config['train_steps'],
        dataset=dataset,
        scene_bound=config['scene_bound'],
        diagonal_n_steps=config['diagonal_n_steps'],
        stepsize_portion=config['stepsize_portion'],
        occupancy_grid=occupancy_grid,
        state=state
    )
    
    state, occupancy_grid = train_loop_with_args()

    render_fn = partial(
        render_scene,
        # Patch size has to be small otherwise not all rays will produce samples and the
        # resulting image will have artifacts. This can be fixed by switching to the 
        # inference version of the ray marching and ray integration functions.
        patch_size_x=32,
        patch_size_y=32,
        dataset=dataset,
        scene_bound=config['scene_bound'],
        diagonal_n_steps=config['diagonal_n_steps'],
        grid_cascades=1,
        grid_resolution=config['grid_resolution'],
        stepsize_portion=config['stepsize_portion'],
        occupancy_bitfield=occupancy_grid.bitfield,
        batch_size=config['batch_size'],
        state=state
    )
    render_fn(
        transform_matrix=dataset.transform_matrices[3],
        file_name='ngp_nerf_cuda_rendered_image'
    )
    turntable_render(
        num_frames=60*3,
        camera_distance=1,
        render_fn=render_fn,
        file_name='ngp_nerf_cuda_turntable_render'
    )

if __name__ == '__main__':
    main()