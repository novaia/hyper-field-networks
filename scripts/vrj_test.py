# This is a test of the volume-rendering-jax extension.

# IMPORTANT NOTE: jaxngp doesn't shift translations by 0.5 at load time.
# Instead, it samples in [-bound, bound] and applies the following transformation
# before hash encoding: pos = (pos + bound) / (2 * bound) 
# Currently I'm shifting translations by 0.5 and also sampling in [-bound, bound],
# so I need to change one of these things in order to get the correct results.

import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from fields import ngp_nerf, Dataset
from volrendjax import integrate_rays, integrate_rays_inference, march_rays, march_rays_inference
from volrendjax import morton3d_invert, packbits
from dataclasses import dataclass
import dataclasses
from typing import Callable, List, Literal, Tuple, Type, Union
from flax import struct
import numpy as np
from functools import partial
import optax

@dataclass
class OccupancyGrid:
    # This class is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)s
    # float32, full-precision density values
    density: jax.Array
    # bool, a non-compact representation of the occupancy bitfield
    occ_mask: jax.Array
    # uint8, each bit is an occupancy value of a grid cell
    occupancy: jax.Array
    # uint32, indices of the grids that are alive (trainable)
    alive_indices: jax.Array
    # list of `int`s, upper bound of each cascade
    alive_indices_offset: List[int]=struct.field(pytree_node=False)

    @classmethod
    def create(cls, cascades: int, grid_resolution: int=128):
        """
        Inputs:
            cascades: number of cascades, paper: 𝐾 = 1 for all synthetic NeRF scenes (single grid)
                      and 𝐾 ∈ [1, 5] for larger real-world scenes (up to 5 grids, depending on scene
                      size)
            grid_resolution: resolution of the occupancy grid, the NGP paper uses 128.

        Example usage:
            ogrid = OccupancyDensityGrid.create(cascades=5, grid_resolution=128)
        """
        G3 = grid_resolution**3
        n_grids = cascades * G3
        occupancy = 255 * jnp.ones(
            shape=(n_grids // 8,),  # each bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.float32,
        )
        occ_mask = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.bool_,
        )
        return cls(
            density=density,
            occ_mask=occ_mask,
            occupancy=occupancy,
            alive_indices=jnp.arange(n_grids, dtype=jnp.uint32),
            alive_indices_offset=np.cumsum([0] + [G3] * cascades).tolist(),
        )
    
    def mean_density_up_to_cascade(self, cas: int) -> Union[float, jax.Array]:
        return self.density[self.alive_indices[:self.alive_indices_offset[cas]]].mean()

def update_occupancy_grid_density(
    KEY, cas: int, update_all: bool, max_inference: int, 
    occupancy_grid: OccupancyGrid, density_grid_res: int,
    scene_bound: float, state: TrainState
):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    G3 = density_grid_res**3
    cas_slice = slice(cas * G3, (cas + 1) * G3)
    cas_alive_indices = occupancy_grid.alive_indices[
        occupancy_grid.alive_indices_offset[cas]:occupancy_grid.alive_indices_offset[cas+1]
    ]
    aligned_indices = cas_alive_indices % G3  # values are in range [0, G3)
    n_grids = aligned_indices.shape[0]

    decay = .95
    cas_occ_mask = occupancy_grid.occ_mask[cas_slice]
    cas_density_grid = occupancy_grid.density[cas_slice].at[aligned_indices].set(
        occupancy_grid.density[cas_slice][aligned_indices] * decay
    )

    if update_all:
        # During the first 256 training steps, we sample M = K * 128^{3} cells uniformly without
        # repetition.
        cas_updated_indices = aligned_indices
    else:
        M = max(1, n_grids // 2)
        # The first M/2 cells are sampled uniformly among all cells.
        KEY, key_firsthalf, key_secondhalf = jax.random.split(KEY, 3)
        indices_firsthalf = jax.random.choice(
            key=key_firsthalf,
            a=aligned_indices,
            shape=(max(1, M//2),),
            replace=True,  # allow duplicated choices
        )
        # Rejection sampling is used for the remaining samples to restrict selection to cells
        # that are currently occupied.
        # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
        cas_alive_occ_mask = cas_occ_mask[aligned_indices]
        indices_secondhalf = jax.random.choice(
            key=key_secondhalf,
            a=aligned_indices,
            shape=(max(1, M//2),),
            replace=True,  # allow duplicated choices
            p=cas_alive_occ_mask.astype(jnp.float32),  # only care about occupied grids
        )
        cas_updated_indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

    coordinates = morton3d_invert(cas_updated_indices).astype(jnp.float32)
    coordinates = coordinates / (density_grid_res - 1) * 2 - 1  # in [-1, 1]
    mip_bound = min(scene_bound, 2**cas)
    half_cell_width = mip_bound / density_grid_res
    coordinates *= mip_bound - half_cell_width  # in [-mip_bound+half_cell_width, mip_bound-half_cell_width]
    # random point inside grid cells
    KEY, key = jax.random.split(KEY, 2)
    coordinates += jax.random.uniform(
        key,
        coordinates.shape,
        coordinates.dtype,
        minval=-half_cell_width,
        maxval=half_cell_width,
    )

    '''
    new_densities = map(
        lambda coords_part: jax.jit(jax.vmap(state.apply_fn, in_axes=(None, 0)))(
            {"params": jax.lax.stop_gradient(state.params)},
            (coords_part, coords_part) # Use coords_part as dummy direction.
        )[0].ravel(),
        jnp.array_split(jax.lax.stop_gradient(coordinates), max(1, n_grids // (max_inference))),
    )
    '''
    def compute_density(coord):
        # Use coord as dummy direction.
        drgbs = state.apply_fn({'params': jax.lax.stop_gradient(state.params)}, (coord, coord))
        density = jnp.ravel(drgbs[0])
        return density
    compute_densities = jax.jit(jax.vmap(compute_density, in_axes=0))
    num_splits = max(1, n_grids//max_inference)
    split_coordinates = jnp.array_split(jax.lax.stop_gradient(coordinates), num_splits)
    new_densities = map(compute_densities, split_coordinates)
    new_densities = jnp.ravel(jnp.concatenate(list(new_densities)))

    cas_density_grid = cas_density_grid.at[cas_updated_indices].set(
        jnp.maximum(cas_density_grid[cas_updated_indices], new_densities)
    )
    new_occupancy_grid = dataclasses.replace(
        occupancy_grid, density=occupancy_grid.density.at[cas_slice].set(cas_density_grid)
    )
    return new_occupancy_grid

def density_threshold_from_min_step_size(diagonal_n_steps, scene_bound) -> float:
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    return .01 * diagonal_n_steps / (2 * min(scene_bound, 1) * 3**.5)

#@jax.jit
def threshold_occupancy_grid(occupancy_grid: OccupancyGrid, diagonal_n_steps, scene_bound):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    mean_density = occupancy_grid.mean_density_up_to_cascade(1)
    density_threshold = jnp.minimum(
        density_threshold_from_min_step_size(diagonal_n_steps, scene_bound), mean_density
    )
    occupied_mask, occupancy_bitfield = packbits(
        density_threshold=density_threshold,
        density_grid=occupancy_grid.density,
    )
    new_occupancy_grid = dataclasses.replace(
        occupancy_grid,
        occ_mask=occupied_mask,
        occupancy=occupancy_bitfield,
    )
    return new_occupancy_grid

@jax.jit
def get_ray(uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    direction = direction / jnp.linalg.norm(direction)
    origin = transform_matrix[:3, -1]
    return origin, direction

@jax.jit
def make_near_far_from_bound(
    bound: float,
    o: jax.Array,  # [n_rays, 3]
    d: jax.Array,  # [n_rays, 3]
):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    "Calculates near and far intersections with the bounding box [-bound, bound]^3 for each ray."

    # avoid d[j] being zero
    eps = 1e-15
    d = jnp.where(
        jnp.signbit(d),  # True for negatives, False for non-negatives
        jnp.clip(d, None, -eps * jnp.ones_like(d)),  # if negative, upper-bound is -eps
        jnp.clip(d, eps * jnp.ones_like(d)),  # if non-negative, lower-bound is eps
    )

    # [n_rays]
    tx0, tx1 = (
        (-bound - o[:, 0]) / d[:, 0],
        (bound - o[:, 0]) / d[:, 0],
    )
    ty0, ty1 = (
        (-bound - o[:, 1]) / d[:, 1],
        (bound - o[:, 1]) / d[:, 1],
    )
    tz0, tz1 = (
        (-bound - o[:, 2]) / d[:, 2],
        (bound - o[:, 2]) / d[:, 2],
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    # when t_start<0, or t_start>t_end, ray does not intersect with aabb, these cases are handled in
    # the `march_rays` implementation
    t_start = jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start)  # last axis that gose inside the bbox
    t_end = jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end)  # first axis that goes out of the bbox

    t_start = jnp.maximum(0., t_start)

    # [n_rays], [n_rays]
    return t_start, t_end

def sample_pixels(
    rng,
    num_samples:int, 
    image_width:int, 
    image_height:int, 
    num_images:int, 
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
    indices = (image_indices, width_indices, height_indices)
    return indices 

def train_loop(
    batch_size:int, 
    max_num_rays:int,
    training_steps:int, 
    state:TrainState, 
    dataset:Dataset,
    scene_bound:float,
    diagonal_n_steps:int,
    stepsize_portion:float,
    grid_resolution:int,
    grid_cascades:int,
    occupancy_grid_update_interval:int,
    occupancy_grid:OccupancyGrid,
):
    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    num_images = dataset.images.shape[0]
    for step in range(training_steps):
        loss, state = train_step(
            image_width=dataset.w,
            image_height=dataset.h,
            num_images=num_images,
            images=dataset.images,
            transform_matrices=dataset.transform_matrices,
            focal_length_x=dataset.fl_x,
            focal_length_y=dataset.fl_y,
            principal_point_x=dataset.cx,
            principal_point_y=dataset.cy,
            step=step,
            max_num_rays=max_num_rays,
            total_ray_samples=batch_size,
            scene_bound=scene_bound,
            diagonal_n_steps=diagonal_n_steps,
            stepsize_portion=stepsize_portion,
            grid_resolution=grid_resolution,
            grid_cascades=grid_cascades,
            occupancy_grid_occupancy=occupancy_grid.occupancy,
            state=state
        )
        print('Step', step, 'Loss', loss)
        
        if step % occupancy_grid_update_interval == 0 and step > 0:
            print('Updating occupancy grid')
            update_all = (step < 20)
            if update_all: print('Updating all cells')
            occupancy_grid = update_occupancy_grid_density(
                KEY=jax.random.PRNGKey(step),
                cas=0,
                update_all=update_all,
                max_inference=batch_size,
                occupancy_grid=occupancy_grid,
                density_grid_res=grid_resolution,
                scene_bound=scene_bound,
                state=state
            )
            occupancy_grid = threshold_occupancy_grid(
                occupancy_grid, diagonal_n_steps, scene_bound
            )
        #break
    return state, occupancy_grid

@partial(jax.jit, static_argnames=('total_samples'))
def render_rays_train(
    ray_origins,
    ray_directions,
    total_samples:int, 
    scene_bound:float,
    diagnoal_n_steps:int,
    stepsize_portion:float,
    grid_resolution:int,
    grid_cascades:int,
    occupancy_grid_occupancy:jax.Array,
    state:TrainState
):
    t_starts, t_ends = make_near_far_from_bound(
        bound=scene_bound, o=ray_origins, d=ray_directions
    )
    noises = jnp.zeros((ray_origins.shape[0],))

    (measured_batch_size_before_compaction, ray_is_valid, rays_n_samples,
    rays_sample_startidx, ray_idcs, xyzs, dirs, dss, z_vals) = march_rays(
        total_samples=total_samples, 
        diagonal_n_steps=diagnoal_n_steps,
        K=grid_cascades,
        G=grid_resolution,
        bound=scene_bound,
        stepsize_portion=stepsize_portion,
        rays_o=ray_origins,
        rays_d=ray_directions,
        t_starts=jnp.ravel(t_starts),
        t_ends=jnp.ravel(t_ends),
        noises=noises,
        occupancy_bitfield=occupancy_grid_occupancy
    )

    # What is tv?
    # Apparently it means total variation.
    drgbs, tv = state.apply_fn({"params": state.params}, xyzs, dirs)
    
    background_colors = jnp.ones((rays_n_samples, 3))
    effective_samples, final_rgbds, final_opacities = integrate_rays(
        near_distance=0.1,
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=background_colors,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )

    return (
        jnp.sum(ray_is_valid), 
        ray_is_valid, 
        measured_batch_size_before_compaction, 
        jnp.where(effective_samples > 0, effective_samples, 0).sum(),
        final_rgbds
    )

#@partial(jax.jit, static_argnames=(
#    'image_width',
#    'image_height',
#    'num_images',
#    'total_ray_samples',
#    'max_num_rays',
#))
def train_step(
    image_width:int,
    image_height:int,
    num_images:int,
    images:jax.Array,
    transform_matrices:jax.Array,
    focal_length_x:float,
    focal_length_y:float,
    principal_point_x:float,
    principal_point_y:float,
    step:int, 
    max_num_rays:int, 
    total_ray_samples:int,
    scene_bound:float,
    diagonal_n_steps:int,
    stepsize_portion:float,
    grid_resolution:int,
    grid_cascades:int,
    occupancy_grid_occupancy:jax.Array,
    state:TrainState
):
    pixel_sample_key, random_bg_key = jax.random.split(jax.random.PRNGKey(step), num=2)
    image_indices, width_indices, height_indices = sample_pixels(
        pixel_sample_key, max_num_rays, image_width, image_height, num_images,
    )
    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    ray_origins, ray_directions = get_ray_vmap(
        width_indices, height_indices, transform_matrices[image_indices], 
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
        occupancy_bitfield=occupancy_grid_occupancy
    )
    num_valid_rays = jnp.sum(ray_is_valid)
    #print('Num valid rays:', num_valid_rays)

    @jax.jit
    def compute_sample(params, ray_sample, direction):
        return state.apply_fn({'params': params}, (ray_sample, direction))
    compute_batch = jax.vmap(compute_sample, in_axes=(None, 0, 0))

    def loss_fn(params):
        drgbs = compute_batch(params, xyzs, dirs)
        #print('drgbs shape', drgbs.shape)
        background_colors = jnp.ones((max_num_rays, 3))
        #print('background colors shape', background_colors.shape)
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
        target_rgbs = images[image_indices, width_indices, height_indices, :3]
        # Might want to divide only by num_valid_rays in other implementation as well.
        loss = jnp.sum(jnp.where(
            ray_is_valid, 
            jnp.mean(optax.huber_loss(pred_rgbs, target_rgbs, delta=0.1), axis=-1),
            0.0,
        )) / num_valid_rays
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # The gradients for all invalid samples are zero. 
    # The presence of so many zeros introduces numerical instability which 
    # causes there to be NaNs in the gradients.
    # nan_to_num is a quick way to fix this by setting all the NaNs to zero.
    #grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
    state = state.apply_gradients(grads=grads)
    return loss, state

class NGPNerfWithDepth(nn.Module):
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
        encoded_position = ngp_nerf.MultiResolutionHashEncoding(
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
            density = jnp.exp(x[0:1])
        else:
            density = nn.activation.relu(x[0:1])
        density_feature = x

        encoded_direction = ngp_nerf.fourth_order_sh_encoding(direction)
        x = jnp.concatenate([density_feature, jnp.ravel(encoded_direction)], axis=0)
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

if __name__ == '__main__':
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = False

    learning_rate = 1e-2
    epsilon = 1e-15
    weight_decay_coefficient = 1e-6
    ray_near = 0.2
    ray_far = 3.0
    batch_size = 30000
    train_target_samples_per_ray = 32
    train_max_rays = batch_size // train_target_samples_per_ray
    render_max_samples_per_ray = 128
    training_steps = 200
    num_turntable_render_frames = 3
    turntable_render_camera_distance = 1.4
    render_patch_size_x = 32
    render_patch_size_y = 32
    num_density_grid_points = 32

    scene_bound = 0.5
    diagonal_n_steps = 1024
    stepsize_portion = 1.0 / 256.0
    grid_resolution = 128
    grid_cascades = 1
    occupancy_grid_update_interval = 16

    dataset = ngp_nerf.load_dataset('data/lego', 1)
    print('Horizontal FOV:', dataset.horizontal_fov)
    print('Vertical FOV:', dataset.vertical_fov)
    print('Focal length x:', dataset.fl_x)
    print('Focal length y:', dataset.fl_y)
    print('Principal point x:', dataset.cx)
    print('Principal point y:', dataset.cy)
    print('Image width:', dataset.w)
    print('Image height:', dataset.h)
    print('Images shape:', dataset.images.shape)
    print('Max num rays: ', train_max_rays)

    occupancy_grid = OccupancyGrid.create(cascades=1, grid_resolution=128)

    model = NGPNerfWithDepth(
        number_of_grid_levels=num_hash_table_levels,
        max_hash_table_entries=max_hash_table_entries,
        hash_table_feature_dim=hash_table_feature_dim,
        coarsest_resolution=coarsest_resolution,
        finest_resolution=finest_resolution,
        density_mlp_width=density_mlp_width,
        color_mlp_width=color_mlp_width,
        high_dynamic_range=high_dynamic_range,
        exponential_density_activation=exponential_density_activation
    )
    rng = jax.random.PRNGKey(1)
    state = ngp_nerf.create_train_state(
        model=model, rng=rng, learning_rate=learning_rate, 
        epsilon=epsilon, weight_decay_coefficient=weight_decay_coefficient
    )
    del rng
    state, occupancy_grid = train_loop(
        batch_size=batch_size, 
        max_num_rays=train_max_rays, 
        training_steps=training_steps, 
        state=state, 
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        stepsize_portion=stepsize_portion,
        grid_resolution=grid_resolution,
        grid_cascades=grid_cascades,
        occupancy_grid_update_interval=occupancy_grid_update_interval,
        occupancy_grid=occupancy_grid,
    )