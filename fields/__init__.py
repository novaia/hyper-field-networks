from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp
import jax

def straight_alpha_composite(foreground, background, alpha):
    return foreground * alpha + background * (1 - alpha)

def alpha_composite(foreground, background, alpha):
    return foreground + background * (1 - alpha)

@dataclass
class Dataset:
    horizontal_fov: float
    vertical_fov: float
    fl_x: Optional[float] = None # Focal length x.
    fl_y: Optional[float] = None # Focal length y.
    k1: Optional[float] = None # First radial distortion parameter.
    k2: Optional[float] = None # Second radial distortion parameter.
    p1: Optional[float] = None # Third radial distortion parameter.
    p2: Optional[float] = None # Fourth radial distortion parameter.
    cx: Optional[float] = None # Principal point x.
    cy: Optional[float] = None # Principal point y.
    w: Optional[int] = None # Image width.
    h: Optional[int] = None # Image height.
    aabb_scale: Optional[int] = None # Scale of scene bounding box.
    transform_matrices: Optional[jnp.ndarray] = None
    images: Optional[jax.Array] = None

def frequency_encoding(x, min_deg, max_deg):
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    xb = x * scales
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.ravel(jnp.concatenate([x] + [four_feat], axis=-1))

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