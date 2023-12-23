import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

resolutions = [4, 8, 16, 32, 64]
reals = []
fakes = []

for res in resolutions:
    reals.append(jnp.ones((32, 3, res, res)))
    fakes.append(jnp.ones((32, 3, res, res)))
    #print(fakes[-1].shape)

@jax.jit
def concat_real_and_fake(real, fake):
    result = []
    for i in range(len(real)):
        result.append(jnp.concatenate([real[i], fake[i]], axis=0))
    return result

collated = concat_real_and_fake(reals, fakes)
#for c in collated:
#   print(c.shape)

@partial(jax.jit, static_argnames=['levels'])
def make_image_pyramid(image, levels):
    pyramid = [image]
    for _ in range(1, levels):
        last_image = pyramid[-1]
        new_shape = (
            last_image.shape[0], 
            last_image.shape[1], 
            last_image.shape[2]//2, 
            last_image.shape[3]//2
        )
        pyramid.append(jax.image.resize(image, new_shape, method='bilinear'))
    return pyramid

image = jnp.ones((32, 3, 64, 64))
image_pyramid = make_image_pyramid(image, 5)
for i in range(len(image_pyramid)):
    print(image_pyramid[i].shape)