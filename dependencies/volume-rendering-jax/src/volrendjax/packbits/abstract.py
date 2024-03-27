import chex
import jax
from jax import numpy as jnp
from jax.core import ShapedArray

def pack_density_into_bits_abstract(density_threshold: ShapedArray, density_grid: ShapedArray):
    chex.assert_rank([density_threshold, density_grid], 1)
    chex.assert_shape(density_threshold, density_grid.shape)
    n_bits = density_grid.shape[0]
    if n_bits % 8 != 0:
        raise ValueError(
            "pack_density_into_bits expects size of density grid to be "
            "divisible by 8, got {}".format(n_bits)
        )
    n_bytes = n_bits // 8

    dtype = jax.dtypes.canonicalize_dtype(density_grid.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "pack_density_into_bits is only implemented for densities "
            "of `jnp.float32` type, got {}".format(dtype)
        )

    out_shapes = {
        "occupied_mask": (n_bits,),
        "occupancy_bitfield": (n_bytes,),
    }
    return (
        ShapedArray(out_shapes["occupied_mask"], jnp.bool_),
        ShapedArray(out_shapes["occupancy_bitfield"], jnp.uint8),
    )
