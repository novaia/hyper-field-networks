import flax.linen as nn
import jax.numpy as jnp
import jax

# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

# Calculates the fourth order spherical harmonic encoding for the given directions.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
# TODO: vectorize this with jax.vmap
def fourth_order_sh_encoding(directions):
    num_components = 16
    components = jnp.zeros(directions.shape[-1], num_components)

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # This could probably be optimized by calculating the components in a 1x16 array, 
    # then copying it three times to get a 3x16 array. This way, all three axes wouldn't
    # have to be indexed every time.
    components[..., 0] = 0.28209479177387814
    components[..., 1] = 0.4886025119029199 * y
    components[..., 2] = 0.4886025119029199 * z
    components[..., 3] = 0.4886025119029199 * x
    components[..., 4] = 1.0925484305920792 * x * y
    components[..., 5] = 1.0925484305920792 * y * z
    components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
    components[..., 7] = 1.0925484305920792 * x * z
    components[..., 8] = 0.5462742152960396 * (xx - yy)
    components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
    components[..., 10] = 2.890611442640554 * x * y * z
    components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
    components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
    components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
    components[..., 14] = 1.445305721320277 * z * (xx - yy)
    components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    return components

class InstantNerf(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.

    def __call__(self, x):
        return x