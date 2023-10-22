import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
import flax.linen as nn
from fields import ngp_nerf_cuda, Dataset, temp
from volrendjax import morton3d_invert
import numpy as np
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

def test_occupancy_grid_functions(state):
    batch_size = 30000
    scene_bound = 1.0
    diagonal_n_steps = 1024
    grid_resolution = 128
    warmup = False

    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(grid_resolution=grid_resolution)
    occupancy_grid.densities = ngp_nerf_cuda.update_occupancy_grid_density(
        KEY=jax.random.PRNGKey(0),
        batch_size=batch_size,
        densities=occupancy_grid.densities,
        occupancy_mask=occupancy_grid.mask,
        grid_resolution=occupancy_grid.grid_resolution,
        num_grid_entries=occupancy_grid.num_entries,
        scene_bound=scene_bound,
        state=state,
        warmup=warmup,
    )

    print('mask shape before', occupancy_grid.mask.shape)
    print('bitfield shape before', occupancy_grid.bitfield.shape)
    occupancy_grid.mask, occupancy_grid.bitfield = ngp_nerf_cuda.threshold_occupancy_grid(
        diagonal_n_steps=diagonal_n_steps,
        scene_bound=scene_bound,
        densities=occupancy_grid.densities
    )
    print('mask shape after', occupancy_grid.mask.shape)
    print('bitfield shape after', occupancy_grid.bitfield.shape)

def update_occupancy_grid(
    batch_size:int, diagonal_n_steps:int, scene_bound:float, step:int,
    state:ngp_nerf_cuda.TrainState, occupancy_grid:ngp_nerf_cuda.OccupancyGrid
):
    warmup = step < occupancy_grid.warmup_steps
    #occupancy_grid.densities = jax.lax.stop_gradient(
    densities_refactored = jax.lax.stop_gradient(
        ngp_nerf_cuda.update_occupancy_grid_density(
            KEY=jax.random.PRNGKey(step),
            batch_size=batch_size,
            densities=occupancy_grid.densities,
            occupancy_mask=occupancy_grid.mask,
            grid_resolution=occupancy_grid.grid_resolution,
            num_grid_entries=occupancy_grid.num_entries,
            scene_bound=scene_bound,
            state=state,
            warmup=warmup
        )
    )
    densities_original = jax.lax.stop_gradient(
        temp.update_occupancy_grid_density(
            KEY=jax.random.PRNGKey(step),
            cas=0,
            update_all=warmup,
            max_inference=batch_size,
            alive_indices=jnp.arange(occupancy_grid.num_entries, dtype=jnp.uint32),
            alive_indices_offset=jnp.array([0, occupancy_grid.num_entries], dtype=jnp.uint32),
            occ_mask=occupancy_grid.mask,
            density=occupancy_grid.densities,
            density_grid_res=occupancy_grid.grid_resolution,
            scene_bound=scene_bound,
            state=state
        )
    )
    print(densities_refactored)
    print(densities_original)
    assertion_text = 'Densities refactored and original are not equal.'
    #assert jnp.allclose(densities_refactored, densities_original), assertion_text
    occupancy_grid.densities = densities_refactored

    #occupancy_grid.mask, occupancy_grid.bitfield = jax.lax.stop_gradient(
    mask_refactored, bitfield_refactored = jax.lax.stop_gradient(
        ngp_nerf_cuda.threshold_occupancy_grid(
            diagonal_n_steps=diagonal_n_steps,
            scene_bound=scene_bound,
            densities=occupancy_grid.densities
        )
    )
    mask_original, bitfield_original = jax.lax.stop_gradient(
        temp.threshold_occupancy_grid(
            mean_density=jnp.mean(occupancy_grid.densities),
            diagonal_n_steps=diagonal_n_steps,
            scene_bound=scene_bound,
            density=occupancy_grid.densities
        )
    )
    print('masks')
    print(mask_refactored)
    print(mask_original)
    print('bitfields')
    print(bitfield_refactored)
    print(bitfield_original)
    assert(jnp.allclose(mask_refactored, mask_original)), 'masks not close'
    assert(jnp.allclose(bitfield_refactored, bitfield_original)), 'bitfields not close'
    occupancy_grid.mask, occupancy_grid.bitfield = mask_refactored, bitfield_refactored
    return occupancy_grid

def train_loop(
    batch_size:int, 
    train_steps:int, 
    dataset:Dataset, 
    scene_bound:float, 
    diagonal_n_steps:int, 
    stepsize_portion:float,
    occupancy_grid:ngp_nerf_cuda.OccupancyGrid,
    state:ngp_nerf_cuda.TrainState
):
    num_images = dataset.images.shape[0]
    for step in range(train_steps):
        loss, state = ngp_nerf_cuda.train_step(
            KEY=jax.random.PRNGKey(step),
            batch_size=batch_size,
            image_width=dataset.w,
            image_height=dataset.h,
            num_images=num_images,
            images=dataset.images,
            transform_matrices=dataset.transform_matrices,
            state=state,
            occupancy_bitfield=occupancy_grid.bitfield,
            grid_resolution=occupancy_grid.grid_resolution,
            principal_point_x=dataset.cx,
            principal_point_y=dataset.cy,
            focal_length_x=dataset.fl_x,
            focal_length_y=dataset.fl_y,
            scene_bound=scene_bound,
            diagonal_n_steps=diagonal_n_steps,
            stepsize_portion=stepsize_portion
        )
        print('Step', step, 'Loss', loss)

        if step % occupancy_grid.update_interval == 0 and step > 0:
            print('Updating occupancy grid...')
            occupancy_grid = update_occupancy_grid(
                batch_size=batch_size,
                diagonal_n_steps=diagonal_n_steps,
                scene_bound=scene_bound,
                step=step,
                state=state,
                occupancy_grid=occupancy_grid
            )
    return state, occupancy_grid

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

    model = ngp_nerf_cuda.NGPNerf(
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
    state = ngp_nerf_cuda.create_train_state(
        model=model, 
        rng=state_init_key, 
        learning_rate=learning_rate,
        epsilon=epsilon,
        weight_decay_coefficient=weight_decay_coefficient
    )
    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(
        grid_resolution=grid_resolution, 
        update_interval=grid_update_interval, 
        warmup_steps=grid_warmup_steps
    )
    dataset = ngp_nerf_cuda.load_dataset('data/lego', 1)
    state, occupancy_grid = train_loop(
        batch_size=batch_size,
        train_steps=train_steps,
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        stepsize_portion=stepsize_portion,
        occupancy_grid=occupancy_grid,
        state=state
    )
    jnp.save('data/occupancy_grid_density.npy', occupancy_grid.mask.astype(np.float32))
    occupancy_grid_coordinates = morton3d_invert(
        jnp.arange(occupancy_grid.mask.shape[0], dtype=jnp.uint32)
    )
    occupancy_grid_coordinates = occupancy_grid_coordinates / (grid_resolution - 1) * 2 - 1
    jnp.save('data/occupancy_grid_coordinates.npy', occupancy_grid_coordinates)

if __name__ == '__main__':
    main()