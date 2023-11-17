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
from fields.common.nn import FrequencyEncoding, FeedForward
import os
import json
import argparse

class ImageField(nn.Module):
    mlp_depth:int
    mlp_width:int
    encoding_dim:int

    @nn.compact
    def __call__(self, x):
        x = FrequencyEncoding(0, self.encoding_dim)(jnp.expand_dims(x, axis=-1))
        x = FeedForward(
            num_layers=self.mlp_depth, hidden_dim=self.mlp_width, 
            output_dim=3, activation=nn.relu
        )(x)
        x = nn.activation.sigmoid(x)
        return x

def train_loop(image, batch_size, steps, state, return_final_loss):
    image_height = image.shape[0]
    image_width = image.shape[1]
    for step in range(steps):
        rng = jax.random.PRNGKey(step)
        loss, state = train_step(image, image_width, image_height, batch_size, state, rng)
    if return_final_loss:
        return state, loss
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

def render_image(state, image_width, image_height):
    x = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, 1, image_width),
        jnp.linspace(0, 1, image_height)
    ), axis=-1)
    x = jnp.reshape(x, (image_width * image_height, 2))
    image = state.apply_fn({'params': state.params}, x)
    image = jnp.reshape(image, (image_width, image_height, 3))
    image = jnp.clip(image, 0, 1)
    image = jnp.transpose(image, (1, 0, 2))
    return image

def create_model_from_config(config:dict):
    model = ImageField(
        mlp_depth=config['mlp_depth'], 
        mlp_width=config['mlp_width'], 
        encoding_dim=config['encoding_dim']
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/image_field.json')
    parser.add_argument('--input_path', type=str, default='data/CIFAR10/testairplane0596.jpg')
    args = parser.parse_args()
    assert os.path.isfile(args.input_path)

    image_path = 'data/CIFAR10/testairplane0596.jpg'
    image = Image.open(image_path)
    image = jnp.array(image) / 255.0

    with open(args.config) as f:
        config = json.load(f)

    model = create_model_from_config(config)
    state = create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    state = train_loop(image, image.shape[0], image.shape[1], 32, 1000, state)
    render_image(state, image.shape[0], image.shape[1], 'data/image_field_output.png')

if __name__ == '__main__':
    main()