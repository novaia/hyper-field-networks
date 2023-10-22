import functools
import random
from typing import Any, Dict, Hashable, Iterable, Sequence, get_args, Union

import jax
from jax._src.lib import xla_client as xc
import jax.random as jran
import numpy as np
import tensorflow as tf

def compose(*fns):
    def _inner(x):
        for fn in fns:
            x = fn(x)
        return x
    return _inner

def mkValueError(desc, value, type):
    variants = get_args(type)
    assert value not in variants
    return ValueError("Unexpected {}: '{}', expected one of [{}]".format(desc, value, "|".join(variants)))

# NOTE:
#   Jitting a vmapped function seems to give the desired performance boost, while vmapping a jitted
#   function might not work at all.  Except for the experiments I conducted myself, some related
#   issues:
# REF:
#   * <https://github.com/google/jax/issues/6312>
#   * <https://github.com/google/jax/issues/7449>
def vmap_jaxfn_with(
        # kwargs copied from `jax.vmap` source
        in_axes: Union[int, Sequence[Any]]=0,
        out_axes: Any = 0,
        axis_name: Union[Hashable, None] = None,
        axis_size: Union[int, None] = None,
        spmd_axis_name: Union[Hashable, None] = None,
    ):
    return functools.partial(
        jax.vmap,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )

def jit_jaxfn_with(
        # kwargs copied from `jax.jit` source
        static_argnums: Union[int, Iterable[int], None] = None,
        static_argnames: Union[str, Iterable[str], None] = None,
        device: Union[xc.Device, None] = None,
        backend: Union[str, None] = None,
        donate_argnums: Union[int, Iterable[int]] = (),
        inline: bool = False,
        keep_unused: bool = False,
        abstracted_axes: Any | None = None,
    ):
    return functools.partial(
        jax.jit,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        device=device,
        backend=backend,
        donate_argnums=donate_argnums,
        inline=inline,
        keep_unused=keep_unused,
        abstracted_axes=abstracted_axes,
    )

def set_deterministic(seed: int) -> jran.KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return jran.PRNGKey(seed)
