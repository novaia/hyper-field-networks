import jax
from jax.core import ShapedArray
import jax.numpy as jnp

def morton3d_abstract(xyzs: ShapedArray):
    length, _ = xyzs.shape

    dtype = jax.dtypes.canonicalize_dtype(xyzs.dtype)
    if dtype != jnp.uint32:
        raise NotImplementedError(
            "morton3d is only implemented for input coordinates " 
            "of type `jnp.uint32`, got {}".format(dtype)
        )

    out_shapes = {
        "idcs": (length,),
    }
    return ShapedArray(shape=out_shapes["idcs"], dtype=jnp.uint32)

def morton3d_invert_abstract(idcs: ShapedArray):
    length, = idcs.shape

    dtype = jax.dtypes.canonicalize_dtype(idcs.dtype)
    if dtype != jnp.uint32:
        raise NotImplementedError(
            "morton3d_invert is only implemented for input indices "
            "of type `jnp.uint32`, got {}".format(dtype)
        )

    out_shapes = {
        "xyzs": (length, 3),
    }
    return ShapedArray(shape=out_shapes["xyzs"], dtype=jnp.uint32)
