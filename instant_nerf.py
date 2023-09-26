import flax.linen as nn
import jax.numpy as jnp
import jax

# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

def hash_function(x, table_size, hash_offset):
    pre_xor = x * jnp.array([1, 2654435761, 805459861])
    x = jnp.bitwise_xor(pre_xor[0], pre_xor[1])
    x = jnp.bitwise_xor(x, pre_xor[2])
    x %= table_size
    x += hash_offset
    return x
vector_hash_function = jax.vmap(hash_function, in_axes=0)

# I made a mistake.

def hash_encoding(x, hash_table, table_size):
    num_levels = 16
    coarsest_resolution = 16
    finest_resolution = 1024
    # Repeat table_size so it can be passed into vectorized hash function.
    table_size_repeated = jnp.repeat(table_size, num_levels)

    levels = jnp.arange(num_levels)
    hash_offset = levels * table_size
    hash_offset_reshaped = jnp.reshape(hash_offset, (hash_offset.shape[0], 1))
    growth_factor = jnp.exp(
        (jnp.log(finest_resolution) - jnp.log(coarsest_resolution)) / (num_levels - 1)
    ) if num_levels > 1 else 1
    scalings = jnp.floor(coarsest_resolution * growth_factor**levels)
    scalings = jnp.reshape(scalings, (scalings.shape[0], 1))

    scaled = x * scalings
    scaled_c = jnp.ceil(scaled).astype(jnp.int32)
    scaled_f = jnp.floor(scaled).astype(jnp.int32)
    
    vertex_0 = scaled_c
    vertex_1 = jnp.concatenate([scaled_c[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
    vertex_2 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
    vertex_3 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_c[:, 2:3]], axis=-1)
    vertex_4 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)
    vertex_5 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
    vertex_6 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
    vertex_7 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)

    hashed_0 = vector_hash_function(vertex_0, table_size, hash_offset_reshaped)
    hashed_1 = vector_hash_function(vertex_1, table_size, hash_offset_reshaped)
    hashed_2 = vector_hash_function(vertex_2, table_size, hash_offset_reshaped)
    hashed_3 = vector_hash_function(vertex_3, table_size, hash_offset_reshaped)
    hashed_4 = vector_hash_function(vertex_4, table_size, hash_offset_reshaped)
    hashed_5 = vector_hash_function(vertex_5, table_size, hash_offset_reshaped)
    hashed_6 = vector_hash_function(vertex_6, table_size, hash_offset_reshaped)
    hashed_7 = vector_hash_function(vertex_7, table_size, hash_offset_reshaped)

    f_0 = hash_table[hashed_0]
    f_1 = hash_table[hashed_1]
    f_2 = hash_table[hashed_2]
    f_3 = hash_table[hashed_3]
    f_4 = hash_table[hashed_4]
    f_5 = hash_table[hashed_5]
    f_6 = hash_table[hashed_6]
    f_7 = hash_table[hashed_7]

    # Linearly interpolate between the eight features here.
    # ...

    return f_0

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