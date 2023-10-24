import jax
import jax.numpy as jnp
import functools
import flax.linen as nn
from flax.training.train_state import TrainState
from volrendjax import morton3d_invert, packbits, march_rays, integrate_rays

# LICENSE: ../dependencies/volume-rendering-jax/LICENSE

# TODO: move this into the volume-rendering-jax library.
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

def truncated_exponential():
    @jax.custom_vjp
    def trunc_exp(x):
        "Exponential function, except its gradient calculation uses a truncated input value"
        return jnp.exp(x)
    def __fwd_trunc_exp(x):
        y = trunc_exp(x)
        aux = x  # aux contains additional information that is useful in the backward pass
        return y, aux
    def __bwd_trunc_exp(aux, grad_y):
        # REF: <https://github.com/NVlabs/instant-ngp/blob/d0d35d215c7c63c382a128676f905ecb676fa2b8/src/testbed_nerf.cu#L303>
        grad_x = jnp.exp(jnp.clip(aux, -15, 15)) * grad_y
        return (grad_x, )
    trunc_exp.defvjp(
        fwd=__fwd_trunc_exp,
        bwd=__bwd_trunc_exp,
    )
    return trunc_exp

def thresholded_exponential():
    def thresh_exp(x, thresh):
        """
        Exponential function translated along -y direction by 1e-2, and thresholded to have
        non-negative values.
        """
        # paper:
        #   the occupancy grids ... is updated every 16 steps ... corresponds to thresholding
        #   the opacity of a minimal ray marching step by 1 − exp(−0.01) ≈ 0.01
        return nn.relu(jnp.exp(x) - thresh)
    return functools.partial(thresh_exp, thresh=1e-2)

def update_occupancy_grid_density(
    KEY, cas:int, update_all:bool, max_inference:int, 
    alive_indices:jax.Array, occ_mask:jax.Array, density:jax.Array, density_grid_res: int,
    alive_indices_offset:jax.Array, scene_bound:float, state:TrainState
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
        print('cas_updated_indices', cas_updated_indices.shape)
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
    new_density = density.at[cas_slice].set(cas_density_grid)
    return new_density

#@jax.jit
def density_threshold_from_min_step_size(diagonal_n_steps, scene_bound) -> float:
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    # Changed from jaxngp implementation.
    # Old: return .01 * diagonal_n_steps / (2 * min(scene_bound, 1) * 3**.5)
    # TODO: pretty sure the new version does the same thing, but should probably verify.
    #return .01 * diagonal_n_steps / (2 * jnp.minimum(scene_bound, 1) * 3**.5)
    return .01 * diagonal_n_steps / (2 * min(scene_bound, 1) * 3**.5)

#@jax.jit
def threshold_occupancy_grid(mean_density, diagonal_n_steps, scene_bound, density:jax.Array):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    # NOTE: mean_density should equal occupancy_grid.mean_density_up_to_cascade(1).
    density_threshold = jnp.minimum(
        density_threshold_from_min_step_size(diagonal_n_steps, scene_bound), mean_density
    )
    new_occupied_mask, new_occupancy_bitfield = packbits(
        density_threshold=density_threshold,
        density_grid=density,
    )
    return new_occupied_mask, new_occupancy_bitfield

'''
@jax.jit
def threshold_ogrid(self) -> "NeRFState":
    mean_density = self.ogrid.mean_density_up_to_cascade(1)
    density_threshold = jnp.minimum(self.density_threshold_from_min_step_size, mean_density)
    occupied_mask, occupancy_bitfield = packbits(
        density_threshold=density_threshold,
        density_grid=self.ogrid.density,
    )
    new_ogrid = self.ogrid.replace(
        occ_mask=occupied_mask,
        occupancy=occupancy_bitfield,
    )
    return self.replace(ogrid=new_ogrid)
'''

'''
@property
def density_threshold_from_min_step_size(self) -> float:
    return .01 * self.raymarch.diagonal_n_steps / (2 * min(self.scene_meta.bound, 1) * 3**.5)
'''

def render_rays_train(
    KEY, o_world, d_world, bg, total_samples, state, occupancy_bitfield, params
):
    t_starts, t_ends = make_near_far_from_bound(
        bound=1.0, o=o_world, d=d_world
    )

    KEY, noise_key = jax.random.split(KEY, 2)
    noises = jax.random.uniform(
        noise_key, shape=t_starts.shape, dtype=t_starts.dtype, minval=0.0, maxval=1.0
    )

    measured_batch_size_before_compaction, ray_is_valid, rays_n_samples, \
    rays_sample_startidx, ray_idcs, xyzs, dirs, dss, z_vals = march_rays(
        total_samples=total_samples,
        diagonal_n_steps=1024,
        K=1,
        G=128,
        bound=1.0,
        stepsize_portion=1.0/256.0,
        rays_o=o_world,
        rays_d=d_world,
        t_starts=t_starts.ravel(),
        t_ends=t_ends.ravel(),
        noises=noises,
        occupancy_bitfield=occupancy_bitfield,
    )

    #'''
    def compute(xyzs, dirs):
        return state.apply_fn({"params": params}, (xyzs, dirs))
    compute_batch = jax.vmap(compute, in_axes=(0, 0))
    drgbs = compute_batch(xyzs, dirs)
    #'''

    #drgbs, tv = state.nerf_fn({"params": state.params}, xyzs, dirs)
    
    effective_samples, final_rgbds, final_opacities = integrate_rays(
        #near_distance=state.options.camera.near,
        near_distance=0.3,
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=bg,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )

    batch_metrics = {
        "n_valid_rays": ray_is_valid.sum(),
        "ray_is_valid": ray_is_valid,
        "measured_batch_size_before_compaction": measured_batch_size_before_compaction,
        "measured_batch_size": jnp.where(effective_samples > 0, effective_samples, 0).sum(),
    }
    return batch_metrics, final_rgbds