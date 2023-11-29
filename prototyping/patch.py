from PIL import Image
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from functools import partial

image = Image.open('data/ngp_image_old.png')
image = jnp.array(image)[..., :-1]

@partial(jax.jit, static_argnames=['patch_size'])
def patchify(x, patch_size):
    def get_patch(x, horizontal_id, vertical_id):
        start_indices = [horizontal_id, vertical_id, 0]
        slice_sizes = [patch_size, patch_size, 3]
        return lax.dynamic_slice(x, start_indices, slice_sizes)

    num_patches_across = (x.shape[1] // patch_size)
    num_patches_total = num_patches_across**2 
    indices = jnp.arange(num_patches_across) * patch_size
    horizontal_indices, vertical_indices = jnp.meshgrid(indices, indices)
    patches = jax.vmap(
        jax.vmap(get_patch, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
    )(x, horizontal_indices, vertical_indices)
    patches = jnp.reshape(patches, (num_patches_total, patch_size, patch_size, 3))
    return patches, num_patches_across

def depatchify(x, num_patches_across):
    x = jnp.array(jnp.split(x, num_patches_across, axis=0))
    x = jax.vmap(lambda a: jnp.concatenate(a, axis=0), in_axes=0)(x)
    x = jnp.concatenate(x, axis=1)
    return x

patches, num_patches_across = patchify(image, 8)
print('patches', patches.shape)
for i in range(patches.shape[0]):
    plt.imsave(f'data/patches/{i}.png', patches[i])

reconstructed = depatchify(patches, num_patches_across)
print('num patches across', num_patches_across)
print('reconstructed', reconstructed.shape)
plt.imsave('data/patches/reconstructed.png', reconstructed)