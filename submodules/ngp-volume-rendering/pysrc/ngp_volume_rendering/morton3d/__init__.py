import jax
from ngp_volume_rendering.morton3d import impl

def morton3d(xyzs: jax.Array):
    return impl.morton3d_p.bind(xyzs)

def morton3d_invert(idcs: jax.Array):
    return impl.morton3d_invert_p.bind(idcs)
