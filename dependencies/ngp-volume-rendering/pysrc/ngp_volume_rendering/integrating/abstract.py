import chex
import jax
from jax import numpy as jnp
from jax.core import ShapedArray

def integrate_rays_abstract(
    rays_sample_start_idx: jax.Array, rays_n_samples: jax.Array, bgs: jax.Array,
    dss: jax.Array, z_vals: jax.Array, drgbs: jax.Array,
):
    n_rays, = rays_sample_start_idx.shape
    total_samples, = dss.shape

    chex.assert_shape([rays_sample_start_idx, rays_n_samples], (n_rays,))
    chex.assert_shape(bgs, (n_rays, 3))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(drgbs, (total_samples, 4))

    dtype = jax.dtypes.canonicalize_dtype(drgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input prediction (density, color) "
            "of `jnp.float32` type, got {}".format(dtype)
        )

    shapes = {
        "helper.measured_batch_size": (1,),
        "out.final_rgbds": (n_rays, 4),
        "out.final_opacities": (n_rays,),
    }
    return (
        ShapedArray(shape=shapes["helper.measured_batch_size"], dtype=jnp.uint32),
        ShapedArray(shape=shapes["out.final_rgbds"], dtype=jnp.float32),
        ShapedArray(shape=shapes["out.final_opacities"], dtype=jnp.float32),
    )

def integrate_rays_backward_abstract(
    rays_sample_start_idx: jax.Array, rays_n_samples: jax.Array,
    bgs: jax.Array, dss: jax.Array, z_vals: jax.Array, drgbs: jax.Array,
    final_rgbds: jax.Array, final_opacities: jax.Array,
    dL_dfinal_rgbds: jax.Array,
    near_distance: float,
):
    n_rays, = rays_sample_start_idx.shape
    total_samples, = dss.shape

    chex.assert_shape([rays_sample_start_idx, rays_n_samples, final_opacities], (n_rays,))
    chex.assert_shape(bgs, (n_rays, 3))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(drgbs, (total_samples, 4))
    chex.assert_shape([final_rgbds, dL_dfinal_rgbds], (n_rays, 4))

    chex.assert_scalar_non_negative(near_distance)

    dtype = jax.dtypes.canonicalize_dtype(drgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input color "
            "of `jnp.float32` type, got {}".format(dtype)
        )

    out_shapes = {
        "dL_dbgs": (n_rays, 3),
        "dL_dz_vals": (total_samples,),
        "dL_ddrgbs": (total_samples, 4),
    }
    return (
        ShapedArray(shape=out_shapes["dL_dbgs"], dtype=jnp.float32),
        ShapedArray(shape=out_shapes["dL_dz_vals"], dtype=jnp.float32),
        ShapedArray(shape=out_shapes["dL_ddrgbs"], dtype=jnp.float32),
    )

def integrate_rays_inference_abstract(
    rays_bg: ShapedArray, rays_rgbd: ShapedArray, rays_T: ShapedArray, n_samples: ShapedArray,
    indices: ShapedArray, dss: ShapedArray, z_vals: ShapedArray, drgbs: ShapedArray,
):
    n_total_rays, _ = rays_rgbd.shape
    n_rays, march_steps_cap = dss.shape

    chex.assert_shape(rays_bg, (n_total_rays, 3))
    chex.assert_shape(rays_rgbd, (n_total_rays, 4))
    chex.assert_shape(rays_T, (n_total_rays,))
    chex.assert_shape([n_samples, indices], (n_rays,))
    chex.assert_shape([dss, z_vals], (n_rays, march_steps_cap))
    chex.assert_shape(drgbs, (n_rays, march_steps_cap, 4))

    out_shapes = {
        "terminate_cnt": (1,),
        "terminated": (n_rays,),
        "rays_rgbd": (n_rays, 4),
        "rays_T": (n_rays,),
    }
    return (
        ShapedArray(shape=out_shapes["terminate_cnt"], dtype=jnp.uint32),
        ShapedArray(shape=out_shapes["terminated"], dtype=jnp.bool_),
        ShapedArray(shape=out_shapes["rays_rgbd"], dtype=jnp.float32),
        ShapedArray(shape=out_shapes["rays_T"], dtype=jnp.float32),
    )
