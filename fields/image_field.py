# A simple neural field for approximating images.
# This will be used to test the feasibility of generating neural field 
# weights with a hypernetwork.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from fields import frequency_encoding

class ImageField(nn.Module):
    mlp_depth:int
    mlp_width:int
    encoding_dim:int

    @nn.compact
    def __call__(self, x):
        x = jax.vmap(frequency_encoding, in_axes=(0, None, None))(
            jnp.expand_dims(x, axis=-1), 0, self.encoding_dim
        )
        for _ in range(self.mlp_depth):
            x = nn.Dense(features=self.mlp_width)(x)
            x = nn.relu(x)
        x = nn.Dense(features=3)(x)
        x = nn.activation.sigmoid(x)
        return x

def train_loop(image, image_width, image_height, batch_size, steps, state):
    for step in range(steps):
        rng = jax.random.PRNGKey(step)
        loss, state = train_step(image, image_width, image_height, batch_size, state, rng)
        #print('Loss:', loss)
    print('Final loss:', loss)
    return state

@partial(jax.jit, static_argnames=('batch_size', 'image_width', 'image_height'))
def train_step(image, image_width, image_height, batch_size, state, rng):
    width_key, height_key = jax.random.split(rng, num=2)
    width_indices = jax.random.randint(
        width_key, shape=(batch_size,), minval=0, maxval=image_width
    )
    height_indices = jax.random.randint(
        height_key, shape=(batch_size,), minval=0, maxval=image_height
    )
    coordinates = jnp.stack([
        width_indices / image_width, height_indices / image_height
    ], axis=-1)
    target_colors = image[width_indices, height_indices, :3]

    def loss_fn(params):
        predicted_colors = state.apply_fn({'params': params}, coordinates)
        loss = jnp.mean(jnp.square(predicted_colors - target_colors))
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def create_train_state(model, learning_rate, rng):
    x = (jnp.ones((1, 2)) / 2)
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def draw_image(state, image_width, image_height, file_path):
    x = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, 1, image_width),
        jnp.linspace(0, 1, image_height)
    ), axis=-1)
    x = jnp.reshape(x, (image_width * image_height, 2))
    image = state.apply_fn({'params': state.params}, x)
    image = jnp.reshape(image, (image_width, image_height, 3))
    image = jnp.clip(image, 0, 1)
    image = jnp.transpose(image, (1, 0, 2))
    plt.imsave(file_path, image)

def main():
    image_path = 'data/approximation_field/approximation_test.png'
    image = Image.open(image_path)
    image = jnp.array(image) / 255.0

    model = ImageField(mlp_depth=3, mlp_width=256, encoding_dim=10)
    state = create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    state = train_loop(image, image.shape[0], image.shape[1], 32, 1000, state)
    draw_image(
        state, image.shape[0], image.shape[1], 'data/approximation_field/approximation.png'
    )