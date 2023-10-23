import jax
import jax.numpy as jnp
from volrendjax import march_rays, integrate_rays

@jax.jit
def make_near_far_from_bound(
    bound: float,
    o: jax.Array,  # [n_rays, 3]
    d: jax.Array,  # [n_rays, 3]
):
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

def render_rays_train(KEY, o_world, d_world, bg, total_samples, state):
    t_starts, t_ends = make_near_far_from_bound(
        bound=state.options.scene_bound, o=o_world, d=d_world
    )

    KEY, noise_key = jax.random.split(KEY, 2)
    noises = jax.random.uniform(
        noise_key, shape=t_starts.shape, dtype=t_starts.dtype, minval=0.0, maxval=1.0
    )

    measured_batch_size_before_compaction, ray_is_valid, rays_n_samples, \
    rays_sample_startidx, ray_idcs, xyzs, dirs, dss, z_vals = march_rays(
        total_samples=total_samples,
        diagonal_n_steps=state.options.diagonal_n_steps,
        K=state.options.cascades,
        G=state.options.density_grid_res,
        bound=state.options.scene_bound,
        stepsize_portion=state.options.stepsize_portion,
        rays_o=o_world,
        rays_d=d_world,
        t_starts=t_starts.ravel(),
        t_ends=t_ends.ravel(),
        noises=noises,
        occupancy_bitfield=state.ogrid.occupancy,
    )

    #'''
    def compute(xyzs, dirs):
        return state.nerf_fn({"params": state.params}, (xyzs, dirs))
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