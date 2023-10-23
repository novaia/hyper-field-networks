import jax
import chex
import jax.numpy as jnp

@jax.jit
def f32_to_u8(img: jax.Array) -> jax.Array:
    return jnp.clip(jnp.round(img * 255), 0, 255).astype(jnp.uint8)

def blend_rgba_image_array(imgarr, bg: jax.Array):
    """
    Blend the given background color according to the given alpha channel from `imgarr`.
    WARN: this function SHOULD NOT be used for blending background colors into volume-rendered
          pixels because the colors of volume-rendered pixels already have the alpha channel
          factored-in.  To blend background for volume-rendered pixels, directly add the scaled
          background color.
          E.g.: `final_color = ray_accumulated_color + (1 - ray_opacity) * bg`
    """
    chex.assert_shape(imgarr, [..., 4])
    chex.assert_type(imgarr, bg.dtype)
    rgbs, alpha = imgarr[..., :-1], imgarr[..., -1:]
    bg = jnp.broadcast_to(bg, rgbs.shape)
    if imgarr.dtype == jnp.uint8:
        rgbs, alpha = rgbs.astype(float) / 255, alpha.astype(float) / 255
        rgbs = rgbs * alpha + bg * (1 - alpha)
        rgbs = f32_to_u8(rgbs)
    else:
        rgbs = rgbs * alpha + bg * (1 - alpha)
    return rgbs
