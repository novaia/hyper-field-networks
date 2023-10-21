import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
import flax.linen as nn
from fields import ngp_nerf_cuda, Dataset
from volrendjax import integrate_rays, integrate_rays_inference, march_rays
from volrendjax import march_rays_inference, morton3d_invert, packbits
import numpy as np
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

def main():
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
    batch_size = 30000
    scene_bound = 1.0

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
    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(grid_resolution=128)
    occupancy_grid.densities = ngp_nerf_cuda.update_occupancy_grid_density(
        KEY=jax.random.PRNGKey(0),
        batch_size=batch_size,
        densities=occupancy_grid.densities,
        occupancy_mask=occupancy_grid.mask,
        grid_resolution=occupancy_grid.grid_resolution,
        num_grid_entries=occupancy_grid.num_entries,
        scene_bound=0.5,
        state=state,
        warmup=True,
    )

if __name__ == '__main__':
    main()