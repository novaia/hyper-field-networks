import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from functools import partial
from typing import Callable, Any

def hash_function_3d(x:jax.Array, table_size:int, hash_offset:jax.Array):
    pre_xor = x * jnp.array([1, 2654435761, 805459861])
    x = jnp.bitwise_xor(pre_xor[..., 0], pre_xor[..., 1])
    x = jnp.bitwise_xor(x, pre_xor[..., 2])
    x = x % table_size
    x = x + hash_offset
    return x

def hash_function_2d(x:jax.Array, table_size:int, hash_offset:jax.Array):
    pre_xor = x * jnp.array([1, 2654435761])
    x = jnp.bitwise_xor(pre_xor[..., 0], pre_xor[..., 1])
    x = x % table_size
    x = x + hash_offset
    return x

def scale_point_to_hash_levels(x:jax.Array, scalings:jax.Array):
    scaled = jnp.einsum('ij,k->ikj', x, scalings)
    scaled_c = jnp.ceil(scaled).astype(jnp.int32)
    scaled_f = jnp.floor(scaled).astype(jnp.int32)
    point_offset = scaled - scaled_f
    return scaled, scaled_c, scaled_f, point_offset

def interpolate_hash_features(feature_a, feature_b, coefficient):
    def _scale(f, c):
        return jnp.einsum('ij,i->ij', f, c)
    return _scale(feature_a, coefficient) + _scale(feature_b, 1-coefficient)

def multi_resolution_hash_encoding_3d(
    x:jax.Array, scalings:jax.Array, hash_offset:jax.Array, table_size:int, hash_table:jax.Array
):
    scaled, scaled_c, scaled_f, point_offset = scale_point_to_hash_levels(x, scalings)
    vertex_0 = scaled_c
    vertex_1 = jnp.concatenate([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], axis=-1)
    vertex_2 = jnp.concatenate([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], axis=-1)
    vertex_3 = jnp.concatenate([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], axis=-1)
    vertex_4 = jnp.concatenate([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], axis=-1)
    vertex_5 = jnp.concatenate([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], axis=-1)
    vertex_6 = jnp.concatenate([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], axis=-1)
    vertex_7 = jnp.concatenate([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], axis=-1)
    
    # First vmap over batch dimension, then vmap over level dimension.
    hash_fn = jax.vmap(
        jax.vmap(
            hash_function_3d, 
            in_axes=(0, None, 0)
        ), 
        in_axes=(0, None, None)
    )
    hashed_0 = hash_fn(vertex_0, table_size, hash_offset)
    hashed_1 = hash_fn(vertex_1, table_size, hash_offset)
    hashed_2 = hash_fn(vertex_2, table_size, hash_offset)
    hashed_3 = hash_fn(vertex_3, table_size, hash_offset)
    hashed_4 = hash_fn(vertex_4, table_size, hash_offset)
    hashed_5 = hash_fn(vertex_5, table_size, hash_offset)
    hashed_6 = hash_fn(vertex_6, table_size, hash_offset)
    hashed_7 = hash_fn(vertex_7, table_size, hash_offset)

    f_0 = hash_table[hashed_0, :]
    f_1 = hash_table[hashed_1, :]
    f_2 = hash_table[hashed_2, :]
    f_3 = hash_table[hashed_3, :]
    f_4 = hash_table[hashed_4, :]
    f_5 = hash_table[hashed_5, :]
    f_6 = hash_table[hashed_6, :]
    f_7 = hash_table[hashed_7, :]

    # Linearly interpolate between all of the features.
    interpolate_fn = jax.vmap(interpolate_hash_features, in_axes=0)
    # First spatial dimension.
    f_03 = interpolate_fn(f_0, f_3, point_offset[..., 0])
    f_12 = interpolate_fn(f_1, f_2, point_offset[..., 0])
    f_56 = interpolate_fn(f_5, f_6, point_offset[..., 0])
    f_47 = interpolate_fn(f_4, f_7, point_offset[..., 0])
    # Second spatial dimension.
    f_0312 = interpolate_fn(f_03, f_12, point_offset[..., 1])
    f_4756 = interpolate_fn(f_47, f_56, point_offset[..., 1])
    # Third spatial dimension.
    encoded = interpolate_fn(f_0312, f_4756, point_offset[..., 3])
    return encoded

def multi_resolution_hash_encoding_2d(
    x:jax.Array, scalings:jax.Array, hash_offset:jax.Array, table_size:int, hash_table:jax.Array
):
    scaled, scaled_c, scaled_f, point_offset = scale_point_to_hash_levels(x, scalings)
    vertex_0 = scaled_c
    vertex_1 = jnp.concatenate([scaled_c[..., 0:1], scaled_f[..., 1:2]], axis=-1)
    vertex_2 = jnp.concatenate([scaled_f[..., 0:1], scaled_c[..., 1:2]], axis=-1)
    vertex_3 = jnp.concatenate([scaled_f[..., 0:1], scaled_f[..., 1:2]], axis=-1)

    # First vmap over batch dimension, then vmap over level dimension.
    hash_fn = jax.vmap(
        jax.vmap(
            hash_function_2d, 
            in_axes=(0, None, 0)
        ), 
        in_axes=(0, None, None)
    )
    hashed_0 = hash_fn(vertex_0, table_size, hash_offset)
    hashed_1 = hash_fn(vertex_1, table_size, hash_offset)
    hashed_2 = hash_fn(vertex_2, table_size, hash_offset)
    hashed_3 = hash_fn(vertex_3, table_size, hash_offset)

    f_0 = hash_table[hashed_0, :]
    f_1 = hash_table[hashed_1, :]
    f_2 = hash_table[hashed_2, :]
    f_3 = hash_table[hashed_3, :]

    # Linearly interpolate between all of the features.
    interpolate_fn = jax.vmap(interpolate_hash_features, in_axes=0)
    # First spatial dimension.
    f_03 = interpolate_fn(f_0, f_3, point_offset[..., 0])
    f_12 = interpolate_fn(f_1, f_2, point_offset[..., 0])
    # Second spatial dimension.
    encoded = interpolate_fn(f_03, f_12, point_offset[..., 1])
    return encoded

class MultiResolutionHashEncoding(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        levels = jnp.arange(self.num_levels)
        self.hash_offset = levels * self.table_size
        if self.num_levels > 1:
            growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            growth_factor = 1
        self.scalings = jnp.floor(self.min_resolution * growth_factor**levels)
        absolute_table_size = self.table_size * self.num_levels
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (absolute_table_size, self.feature_dim)
        )
    
    def __call__(self, x:jax.Array):
        encoded = multi_resolution_hash_encoding_3d(
            x, self.scalings, self.hash_offset, self.table_size, self.hash_table
        )
        return jnp.reshape(encoded, (encoded.shape[0], self.num_levels * self.feature_dim))

class MultiResolutionHashEncoding2D(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.spatial_dim = 2
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        absolute_table_size = self.table_size * self.num_levels
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (absolute_table_size, self.feature_dim)
        )

    def __call__(self, x:jnp.ndarray):
        scaled = jnp.einsum('ij,k->ikj', x, self.scalings)
        scaled_c = jnp.ceil(scaled).astype(jnp.int32)
        scaled_f = jnp.floor(scaled).astype(jnp.int32)
        point_offset = jnp.reshape(scaled - scaled_f, (self.spatial_dim, self.num_levels))
        
        vertex_0 = scaled_c
        vertex_1 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2]], axis=-1)
        vertex_2 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2]], axis=-1)
        vertex_3 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2]], axis=-1)

        hashed_0 = self.hash_function(vertex_0, self.table_size, self.hash_offset)
        hashed_1 = self.hash_function(vertex_1, self.table_size, self.hash_offset)
        hashed_2 = self.hash_function(vertex_2, self.table_size, self.hash_offset)
        hashed_3 = self.hash_function(vertex_3, self.table_size, self.hash_offset)

        f_0 = self.hash_table[:, hashed_0]
        f_1 = self.hash_table[:, hashed_1]
        f_2 = self.hash_table[:, hashed_2]
        f_3 = self.hash_table[:, hashed_3]

        # Linearly interpolate between all of the features.
        f_03 = f_0 * point_offset[0:1, :] + f_3 * (1 - point_offset[0:1, :])
        f_12 = f_1 * point_offset[0:1, :] + f_2 * (1 - point_offset[0:1, :])
        encoded_value = f_03 * point_offset[1:2, :] + f_12 * (1 - point_offset[1:2, :])
        # Transpose so that features are contiguous.
        # i.e. [[f_0_x, f_0_y], [f_1_x, f_2_y], ...]
        # Then ravel to get the entire encoding.
        return jnp.ravel(jnp.transpose(encoded_value))
   
class FeedForward(nn.Module):
    num_layers: int
    hidden_dim: int
    output_dim: int
    activation: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, dtype=self.dtype)(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        return x
    
# Calculates the fourth order spherical harmonic encoding for the given direction.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
def fourth_order_sh_encoding(direction:jnp.ndarray):
    x = direction[0]
    y = direction[1]
    z = direction[2]

    xx = x**2
    yy = y**2
    zz = z**2

    components = jnp.array([
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.9461746957575601 * zz - 0.31539156525251999,
        1.0925484305920792 * x * z,
        0.5462742152960396 * (xx - yy),
        0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * x * y * z,
        0.4570457994644658 * y * (5 * zz - 1),
        0.3731763325901154 * z * (5 * zz - 3),
        0.4570457994644658 * x * (5 * zz - 1),
        1.445305721320277 * z * (xx - yy),
        0.5900435899266435 * x * (xx - 3 * yy)
    ])
    return components

class FrequencyEncoding(nn.Module):
    min_deg: int
    max_deg: int

    @nn.compact
    def __call__(self, x):
        def encode(x):
            scales = jnp.array([2**i for i in range(self.min_deg, self.max_deg)])
            xb = x * scales
            four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
            return jnp.ravel(jnp.concatenate([x] + [four_feat], axis=-1))
        return jax.vmap(encode, in_axes=0)(x)

# Exponential function except its gradient calculation uses a truncated input value.
@jax.custom_vjp
def trunc_exp(x):
    return jnp.exp(x)
def __fwd_trunc_exp(x):
    y = trunc_exp(x)
    aux = x
    return y, aux
def __bwd_trunc_exp(aux, grad_y):
    grad_x = jnp.exp(jnp.clip(aux, -15, 15)) * grad_y
    return (grad_x, )
trunc_exp.defvjp(fwd=__fwd_trunc_exp, bwd=__bwd_trunc_exp)
