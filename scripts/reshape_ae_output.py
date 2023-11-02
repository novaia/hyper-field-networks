import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
import math
from hypernets.packing.ngp_nerf import unpack_weights
from fields import ngp_nerf_cuda
from functools import partial
import json

def render_with_weights(weights, file_name):
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = True
    scene_bound = 1.0
    grid_resolution = 128
    diagonal_n_steps = 1024
    stepsize_portion = 1.0 / 256.0
    batch_size = 256 * 1024
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
    occupancy_grid = ngp_nerf_cuda.create_occupancy_grid(grid_resolution, 0, scene_bound)
    KEY = jax.random.PRNGKey(0)
    state = ngp_nerf_cuda.create_train_state(model, KEY, 1, 1, 1)
    state = state.replace(params=weights)
    dataset = ngp_nerf_cuda.load_dataset('data/synthetic_nerf_data/aliens/alien_0', 1)
    render_fn = partial(
        ngp_nerf_cuda.render_scene,
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
    occupancy_grid.densities = ngp_nerf_cuda.update_occupancy_grid_density(
        KEY, batch_size, occupancy_grid.densities, occupancy_grid.mask, 
        occupancy_grid.resolution, occupancy_grid.num_entries, scene_bound, state, False
    )
    threshold_result = ngp_nerf_cuda.threshold_occupancy_grid(
        diagonal_n_steps, scene_bound, occupancy_grid.densities
    )
    occupancy_grid.mask, occupancy_grid.bitfield = threshold_result
    render_fn(transform_matrix=dataset.transform_matrices[0], file_name=file_name)

def reshape_ae_output(x, square_pad_size, tile_pad_size, table_width):
    num_tiles_per_dim = int(math.ceil(jnp.sqrt(x.shape[0])))
    x = jnp.split(x, num_tiles_per_dim, axis=0)
    vertically_joined = []
    for i in range(len(x)):
        vertically_joined.append(jnp.concatenate(x[i], axis=1))
    x = jnp.concatenate(vertically_joined, axis=0)
    x = jnp.squeeze(x, axis=-1)
    x = x[:-tile_pad_size, :-tile_pad_size]
    x = jnp.ravel(x)[:-square_pad_size]
    table_height = x.shape[0] // table_width
    hash_table = jnp.reshape(x, (table_height, table_width))
    return hash_table, table_height

def main():
    autoencoder_output = jnp.load('data/autoencoder_output.npy', allow_pickle=True).tolist()
    x = autoencoder_output['output']
    square_pad_size = autoencoder_output['square_pad_size']
    tile_pad_size = autoencoder_output['tile_pad_size']
    input_path = autoencoder_output['input_path']
    print('square_pad_size', square_pad_size)
    print('tile_pad_size', tile_pad_size)

    original_packed_weights = jnp.load(input_path)
    hash_table, table_height = reshape_ae_output(x, square_pad_size, tile_pad_size, 64)
    packed_weights = original_packed_weights[table_height:, :]
    packed_weights = jnp.concatenate([hash_table, packed_weights], axis=0)

    with open(os.path.join(os.path.dirname(input_path), 'param_map.json'), 'r') as f:
        weight_map = json.load(f)

    unpacked_weights = unpack_weights(packed_weights, weight_map)
    original_unpacked_weights = unpack_weights(original_packed_weights, weight_map)
    render_with_weights(unpacked_weights, 'ae_output')
    render_with_weights(original_unpacked_weights, 'original')

if __name__ == '__main__':
    main()