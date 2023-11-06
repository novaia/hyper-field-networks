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
    images: Optional[jnp.ndarray] = None