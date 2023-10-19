# This is a test of the volume-rendering-jax extension.
import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from fields import ngp_nerf, Dataset
from volrendjax import integrate_rays, integrate_rays_inference, march_rays, march_rays_inference
from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple, Type
from flax import struct
import numpy as np

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
    occupancy_grid:OccupancyGrid
):
    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    for step in range(training_steps):
        pixel_sample_key, random_bg_key = jax.random.split(jax.random.PRNGKey(step), num=2)
        image_indices, width_indices, height_indices = sample_pixels(
            pixel_sample_key, max_num_rays, dataset.w, dataset.h, dataset.images.shape[0],
        )
        ray_origins, ray_directions = get_ray_vmap(
            width_indices, height_indices, dataset.transform_matrices[image_indices], 
            dataset.cx, dataset.cy, dataset.fl_x, dataset.fl_y
        )
        print(ray_origins.shape, ray_directions.shape)
        t_starts, t_ends = make_near_far_from_bound(bound=0.5, o=ray_origins, d=ray_directions)
        noises = jnp.zeros((ray_origins.shape[0],))
        marched_rays = march_rays(
            total_samples=max_num_rays*batch_size, 
            diagonal_n_steps=1024,
            K=1,
            G=128,
            bound=0.5,
            stepsize_portion=1.0/256.0,
            rays_o=ray_origins,
            rays_d=ray_directions,
            t_starts=jnp.ravel(t_starts),
            t_ends=jnp.ravel(t_ends),
            noises=noises,
            occupancy_bitfield=occupancy_grid.occupancy
        )
        break
    return None

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

    model = ngp_nerf.InstantNerf(
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
    state = train_loop(
        batch_size=batch_size, 
        max_num_rays=train_max_rays, 
        training_steps=training_steps, 
        state=state, 
        dataset=dataset,
        occupancy_grid=occupancy_grid
    )