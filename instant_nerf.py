import flax.linen as nn
import jax.numpy as jnp
import jax

# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

def hash_function(x, table_size):
    pre_xor = x * jnp.array([1, 2654435761, 805459861])
    x = jnp.bitwise_xor(pre_xor[0], pre_xor[1])
    x = jnp.bitwise_xor(x, pre_xor[2])
    x %= table_size
    return x

def hash_encoding(x):
    return x

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

def render_volume(positions, directions, deltas, state):
    assertion_text = 'Positions, directions, and deltas must have the same shape.'
    assert positions.shape == directions.shape == deltas.shape, assertion_text
    densities, colors = state.apply_fn((positions, directions))
    
    triangular_mask = jnp.tril(jnp.ones(densities.shape))
    repeated_densities = jnp.repeat(densities, densities.shape[0], axis=0)
    triangular_densities = repeated_densities * triangular_mask
    repeated_deltas = jnp.repeat(deltas, deltas.shape[0], axis=0)
    triangular_deltas = repeated_deltas * triangular_mask

    T_sum = jnp.exp(-jnp.sum(triangular_densities * triangular_deltas))
    rendered_color = jnp.sum(T_sum * (1 - jnp.exp(-densities * deltas)) * colors)
    return rendered_color

class DensityMLP(nn.Module):
    width: int

    def __call__(self, x):

        return x

class InstantNerf(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.

    density_mlp_width: int
    color_mlp_width: int
    high_dynamic_range: bool

    def __call__(self, x):
        position, direction = x
        encoded_position = hash_encoding(position)

        x = nn.Dense(self.density_mlp_width)(encoded_position)
        x = nn.activation.relu(x)
        density = nn.Dense(16)(x)

        encoded_direction = fourth_order_sh_encoding(direction)
        x = jax.concatenate([density, encoded_direction], axis=0)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(3)(x)

        if self.high_dynamic_range:
            # elu is exponential linear unit, I think that's what the paper meant 
            # by "exponential activation"
            color = nn.activation.elu(x)
        else:
            color = nn.activation.sigmoid(x)

        return density, color