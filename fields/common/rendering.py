import jax
import jax.numpy as jnp

def straight_alpha_composite(foreground:jax.Array, background:jax.Array, alpha:jax.Array):
    return foreground * alpha + background * (1 - alpha)

def alpha_composite(foreground:jax.Array, background:jax.Array, alpha:jax.Array):
    return foreground + background * (1 - alpha)