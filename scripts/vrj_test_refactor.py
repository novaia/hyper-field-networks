import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from fields import ngp_nerf_cuda
from volrendjax import morton3d_invert

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
    state, occupancy_grid = ngp_nerf_cuda.train_loop(
        batch_size=batch_size,
        train_steps=train_steps,
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        stepsize_portion=stepsize_portion,
        occupancy_grid=occupancy_grid,
        state=state
    )
    jnp.save('data/occupancy_grid_density.npy', occupancy_grid.mask.astype(jnp.float32))
    occupancy_grid_coordinates = morton3d_invert(
        jnp.arange(occupancy_grid.mask.shape[0], dtype=jnp.uint32)
    )
    occupancy_grid_coordinates = occupancy_grid_coordinates / (grid_resolution - 1) * 2 - 1
    jnp.save('data/occupancy_grid_coordinates.npy', occupancy_grid_coordinates)

if __name__ == '__main__':
    main()