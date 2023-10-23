import functools
from typing import Callable, List, Tuple, Union

import flax.linen as nn
import jax
from jax.nn.initializers import Initializer
import jax.numpy as jnp

from utils.encoders import (
    Encoder,
    HashGridEncoder,
    SphericalHarmonicsEncoder,
)
from utils.types import empty_impl

@empty_impl
class NeRF(nn.Module):
    bound: float

    position_encoder: Encoder
    direction_encoder: Encoder

    density_mlp: nn.Module
    rgb_mlp: nn.Module

    density_activation: Callable
    rgb_activation: Callable

    @nn.compact
    def __call__(
        self,
        xyz: jax.Array,
        dir: Union[jax.Array, None],
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """
        Inputs:
            xyz `[..., 3]`: coordinates in $\R^3$
            dir `[..., 3]`: **unit** vectors, representing viewing directions.  If `None`, only
                            return densities
            appearance_embeddings `[..., n_extra_learnable_dims]` or `[n_extra_learnable_dims]`:
                per-image latent code to model illumination, if it's a 1D vector of length
                `n_extra_learnable_dims`, all sampled points will use this embedding.

        Returns:
            density `[..., 1]`: density (ray terminating probability) of each query points
            rgb `[..., 3]`: predicted color for each query point
        """
        original_aux_shapes = xyz.shape[:-1]
        n_samples = functools.reduce(int.__mul__, original_aux_shapes)
        xyz = xyz.reshape(n_samples, 3)

        # [n_samples, D_pos], `float32`
        pos_enc, tv = self.position_encoder(xyz, self.bound)

        x = self.density_mlp(pos_enc)
        # [n_samples, 1], [n_samples, density_MLP_out-1]
        density, _ = jnp.split(x, [1], axis=-1)
        density = self.density_activation(density)

        if dir is None:
            return density.reshape(*original_aux_shapes, 1), tv
        dir = dir.reshape(n_samples, 3)

        # [n_samples, D_dir]
        dir_enc = self.direction_encoder(dir)

        # [n_samples, 3]
        rgb = self.rgb_mlp(jnp.concatenate([
            x,
            dir_enc,
        ], axis=-1))
        rgb = self.rgb_activation(rgb)

        return jnp.concatenate([density, rgb], axis=-1).reshape(*original_aux_shapes, 4), tv


class CoordinateBasedMLP(nn.Module):
    "Coordinate-based MLP"

    # hidden layer widths
    Ds: List[int]
    out_dim: int
    skip_in_layers: List[int]

    # as described in the paper
    kernel_init: Initializer=nn.initializers.glorot_uniform()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_x = x
        for i, d in enumerate(self.Ds):
            if i in self.skip_in_layers:
                x = jnp.concatenate([in_x, x], axis=-1)
            x = nn.Dense(
                d,
                use_bias=False,
                kernel_init=self.kernel_init,
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.out_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)
        return x

def make_activation(act):
    if act == "sigmoid":
        return nn.sigmoid
    elif act == "exponential":
        return jnp.exp
    elif act == "truncated_exponential":
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

    elif act == "thresholded_exponential":
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

    elif act == "truncated_thresholded_exponential":
        @jax.custom_vjp
        def trunc_thresh_exp(x, thresh):
            """
            Exponential, but value is translated along -y axis by value `thresh`, negative values
            are removed, and gradient is truncated.
            """
            return nn.relu(jnp.exp(x) - thresh)
        def __fwd_trunc_threash_exp(x, thresh):
            y = trunc_thresh_exp(x, thresh=thresh)
            aux = x, thresh  # aux contains additional information that is useful in the backward pass
            return y, aux
        def __bwd_trunc_threash_exp(aux, grad_y):
            x, thresh = aux
            grad_x = jnp.exp(jnp.clip(x, -15, 15)) * grad_y
            # clip gradient for values that has been thresholded by relu during forward pass
            grad_x = jnp.signbit(jnp.log(thresh) - x) * grad_x
            # first tuple element is gradient for input, second tuple element is gradient for the
            # `threshold` value.
            return (grad_x, 0)
        trunc_thresh_exp.defvjp(
            fwd=__fwd_trunc_threash_exp,
            bwd=__bwd_trunc_threash_exp,
        )
        return functools.partial(trunc_thresh_exp, thresh=1e-2)
    elif act == "relu":
        return nn.relu
    else:
        print('dumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumbdumb')

def make_nerf(
    bound: float,

    # total variation
    tv_scale: float,

    # encoding levels
    pos_levels: int,
    dir_levels: int,

    # layer widths
    density_Ds: List[int],
    rgb_Ds: List[int],

    # output dimensions
    density_out_dim: int,
    rgb_out_dim: int,

    # skip connections
    density_skip_in_layers: List[int],
    rgb_skip_in_layers: List[int],
) -> NeRF:
    position_encoder = HashGridEncoder(
        L=pos_levels,
        T=2**19,
        F=2,
        N_min=2**4,
        N_max=int(2**11 * bound),
        tv_scale=tv_scale,
        param_dtype=jnp.float32,
    )
    direction_encoder = SphericalHarmonicsEncoder(L=dir_levels)
    density_mlp = CoordinateBasedMLP(
        Ds=density_Ds,
        out_dim=density_out_dim,
        skip_in_layers=density_skip_in_layers
    )
    rgb_mlp = CoordinateBasedMLP(
        Ds=rgb_Ds,
        out_dim=rgb_out_dim,
        skip_in_layers=rgb_skip_in_layers
    )

    density_activation = make_activation("truncated_exponential")
    rgb_activation = make_activation("sigmoid")

    model = NeRF(
        bound=bound,

        position_encoder=position_encoder,
        direction_encoder=direction_encoder,

        density_mlp=density_mlp,
        rgb_mlp=rgb_mlp,

        density_activation=density_activation,
        rgb_activation=rgb_activation,
    )

    return model

def make_nerf_ngp(
    bound: float,
    tv_scale: float=0.,
) -> NeRF:
    return make_nerf(
        bound=bound,

        tv_scale=tv_scale,

        pos_levels=16,
        dir_levels=4,

        density_Ds=[64],
        density_out_dim=16,
        density_skip_in_layers=[],

        rgb_Ds=[64, 64],
        rgb_out_dim=3,
        rgb_skip_in_layers=[],
    )