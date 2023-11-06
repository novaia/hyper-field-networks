from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp
import jax

def straight_alpha_composite(foreground, background, alpha):
    return foreground * alpha + background * (1 - alpha)

def alpha_composite(foreground, background, alpha):
    return foreground + background * (1 - alpha)