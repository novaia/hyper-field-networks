import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from fields import ngp_nerf_cuda, Dataset
from fields.temp import make_near_far_from_bound
from volrendjax import march_rays
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rays', type=int, default=20)
    parser.add_argument('--frustum', type=bool, default=False)
    parser.add_argument('--scene_bound', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=30000)
    args = parser.parse_args()

    dataset = ngp_nerf_cuda.load_dataset('data/lego', 1)
    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(128, 0, 0)

    if not args.frustum:
        width_indices = jnp.full((args.num_rays,), dataset.w//2)
        height_indices = jnp.full((args.num_rays,), dataset.h//2)
        image_indices = jnp.arange(args.num_rays)
    else:
        width_indices = jnp.ravel(jnp.repeat(
            jnp.expand_dims((jnp.array([0, 0, dataset.w, dataset.w])), axis=0), 
            args.num_rays, axis=0
        ))
        height_indices = jnp.ravel(jnp.repeat(
            jnp.expand_dims((jnp.array([0, dataset.h, 0, dataset.h])), axis=0), 
            args.num_rays, axis=0
        ))
        image_indices = jnp.ravel(jnp.repeat(
            jnp.expand_dims(jnp.arange(args.num_rays), axis=-1), 4, axis=0
        ))
    transform_matrices = dataset.transform_matrices[image_indices]
    print(width_indices)
    print(height_indices)
    print('width_indices', width_indices.shape)
    print('height_indices', height_indices.shape)
    print('image_indices', image_indices.shape)

    get_rays = jax.vmap(ngp_nerf_cuda.get_ray, in_axes=(0, 0, 0, None, None, None, None))
    ray_origins, ray_directions = get_rays(
        width_indices, height_indices, transform_matrices, dataset.cx, dataset.cy, 
        dataset.fl_x, dataset.fl_y
    )
    print(ray_origins.shape, ray_directions.shape)
    
    t_starts, t_ends = make_near_far_from_bound(args.scene_bound, ray_origins, ray_directions)
    ray_march_result = march_rays(
        total_samples=args.batch_size,
        diagonal_n_steps=128,
        K=1,
        G=occupancy_grid.grid_resolution,
        bound=args.scene_bound,
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
    num_valid_samples = jnp.sum(rays_n_samples)
    print('num_valid_rays', num_valid_rays)
    print('num valid samples', num_valid_samples)
    print('positions shape', positions.shape)
    print('directions shape', directions.shape)
    print(jnp.max(positions))
    print(jnp.min(positions))
    print(rays_n_samples)
    jnp.save('data/ray_marched_positions.npy', positions)
    jnp.save('data/ray_origins.npy', ray_origins)
