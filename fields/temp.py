import jax
import jax.numpy as jnp

# TODO: move this into the volume-rendering-jax library.
@jax.jit
def make_near_far_from_bound(
    bound: float,
    o: jax.Array,  # [n_rays, 3]
    d: jax.Array,  # [n_rays, 3]
):
    # This function is subject to the volume-rendering-jax license 
    # (../dependencies/volume-rendering-jax/LICENSE)
    "Calculates near and far intersections with the bounding box [-bound, bound]^3 for each ray."

    # avoid d[j] being zero
    eps = 1e-15
    d = jnp.where(
        jnp.signbit(d),  # True for negatives, False for non-negatives
        jnp.clip(d, None, -eps * jnp.ones_like(d)),  # if negative, upper-bound is -eps
        jnp.clip(d, eps * jnp.ones_like(d)),  # if non-negative, lower-bound is eps
    )

    # [n_rays]
    tx0, tx1 = (
        (-bound - o[:, 0]) / d[:, 0],
        (bound - o[:, 0]) / d[:, 0],
    )
    ty0, ty1 = (
        (-bound - o[:, 1]) / d[:, 1],
        (bound - o[:, 1]) / d[:, 1],
    )
    tz0, tz1 = (
        (-bound - o[:, 2]) / d[:, 2],
        (bound - o[:, 2]) / d[:, 2],
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    # when t_start<0, or t_start>t_end, ray does not intersect with aabb, these cases are handled in
    # the `march_rays` implementation
    t_start = jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start)  # last axis that gose inside the bbox
    t_end = jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end)  # first axis that goes out of the bbox

    t_start = jnp.maximum(0., t_start)

    # [n_rays], [n_rays]
    return t_start, t_end


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
