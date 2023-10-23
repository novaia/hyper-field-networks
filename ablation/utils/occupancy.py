import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from volrendjax import morton3d_invert
from functools import partial
import optax

# The same occupancy grid update function as vrj_test, but with state.nerf_fn instead of 
# state.apply_fn. This change is only for ablation, so the code was copy-pasted in order
# to not pollute the original code with ablation only logic.
def update_occupancy_grid_density(
    KEY, cas:int, update_all:bool, max_inference:int, 
    alive_indices:jax.Array, occ_mask:jax.Array, density:jax.Array, density_grid_res: int,
    alive_indices_offset:jax.Array, scene_bound:float, state
):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    G3 = density_grid_res**3
    cas_slice = slice(cas * G3, (cas + 1) * G3)
    cas_alive_indices = alive_indices[
        alive_indices_offset[cas]:alive_indices_offset[cas+1]
    ]
    aligned_indices = cas_alive_indices % G3  # values are in range [0, G3)
    n_grids = aligned_indices.shape[0]

    decay = .95
    cas_occ_mask = occ_mask[cas_slice]
    cas_density_grid = density[cas_slice].at[aligned_indices].set(
        density[cas_slice][aligned_indices] * decay
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

    def compute_density(coord):
        # Use coord as dummy direction.
        drgbs = state.nerf_fn({'params': jax.lax.stop_gradient(state.params)}, (coord, coord))
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
    new_density = density.at[cas_slice].set(cas_density_grid)
    return new_density