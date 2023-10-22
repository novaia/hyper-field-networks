import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from fields import ngp_nerf_cuda, Dataset
from fields.temp import make_near_far_from_bound
from volrendjax import march_rays

def main():
    batch_size = 30000
    num_rays = 20
    scene_bound = 1.0
    dataset = ngp_nerf_cuda.load_dataset('data/lego', 1)
    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(128, 0, 0)

    width_indices = jnp.full((num_rays,), dataset.w//2)
    height_indices = jnp.full((num_rays,), dataset.h//2)
    image_indices = jnp.arange(num_rays)
    transform_matrices = dataset.transform_matrices[image_indices]
    
    get_rays = jax.vmap(ngp_nerf_cuda.get_ray, in_axes=(0, 0, 0, None, None, None, None))
    ray_origins, ray_directions = get_rays(
        width_indices, height_indices, transform_matrices, dataset.cx, dataset.cy, 
        dataset.fl_x, dataset.fl_y
    )
    print(ray_origins.shape, ray_directions.shape)
    
    t_starts, t_ends = make_near_far_from_bound(scene_bound, ray_origins, ray_directions)
    ray_march_result = march_rays(
        total_samples=batch_size,
        diagonal_n_steps=128,
        K=1,
        G=occupancy_grid.grid_resolution,
        bound=scene_bound,
        stepsize_portion=1.0 / 256.0,
        rays_o=ray_origins,
        rays_d=ray_directions,
        t_starts=jnp.ravel(t_starts),
        t_ends=jnp.ravel(t_ends),
        noises=0.0,
        occupancy_bitfield=occupancy_grid.bitfield
    )

    _, ray_is_valid, rays_n_samples, rays_sample_start_idx, \
    _, positions, directions, dss, z_vals = ray_march_result
    num_valid_rays = jnp.sum(ray_is_valid)
    print('num_valid_rays', num_valid_rays)
    print('positions shape', positions.shape)
    print('directions shape', directions.shape)
    print(jnp.max(positions))
    print(jnp.min(positions))
    jnp.save('data/ray_marched_positions.npy', positions)

if __name__ == '__main__':
    main()