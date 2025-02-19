import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from fields.common.nn import MultiResolutionHashEncoding, FeedForward
import os, argparse, json
from functools import partial
import optax
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Union, Any
from timeit import default_timer as timer

class NGPImage(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.
    mlp_width: int
    mlp_depth: int
    output_channels: int
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        x = MultiResolutionHashEncoding(
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim,
            spatial_dim=2
        )(x)
        x = FeedForward(
            num_layers=self.mlp_depth,
            hidden_dim=self.mlp_width,
            output_dim=self.output_channels,
            activation=nn.relu,
            dtype=self.dtype
        )(x)
        x = nn.sigmoid(x)
        return x

def train_loop(
    steps:int, state:TrainState, image:jax.Array, batch_size:int, channels:int = 3
) -> TrainState:
    for step in range(steps):
        step_key = jax.random.PRNGKey(step)
        state = train_step(state, image, batch_size, channels)
    return state

def benchmark_train_loop(image, batch_size, steps, state):
    image_height = image.shape[0]
    image_width = image.shape[1]
    channels = image.shape[2]
    
    # Warmup train_step.
    state = train_step(state=state, image=image, batch_size=batch_size)

    start = timer()
    for step in range(steps):
        state = train_step(state=state, image=image, batch_size=batch_size)
    end = timer()
    print('Elapsed time:', end - start)
    start = timer()
    for step in range(steps):
        state = train_step(state=state, image=image, batch_size=batch_size)
    end = timer()
    print('Elapsed time 2:', end - start)

    #print('Final loss:', loss)
    rendered_image = render_image(state, image_width, image_height, channels=channels)
    rendered_image = jnp.array(rendered_image, dtype=jnp.float32)
    plt.imsave('data/ngp_image_benchmark_output.jpg', rendered_image)
    params_cpu = move_pytree_to_cpu(state.params)
    jnp.save(file='data/image_field.npy', arr=params_cpu, allow_pickle=True)

@partial(jax.jit, static_argnames=('batch_size', 'channels'))
def train_step(state:TrainState, image:jax.Array, batch_size:int, channels:int = 3) -> TrainState:
    image_height = image.shape[0]
    image_width = image.shape[1]
    all_height_indices = jnp.arange(image_height)
    all_width_indices = jnp.arange(image_width)
    index_mesh = jnp.reshape(
        jnp.stack(jnp.meshgrid(all_height_indices, all_width_indices), axis=-1), 
        newshape=(-1, 2)
    )

    key = jax.random.PRNGKey(state.step)
    selected_indices = jax.random.choice(key, a=index_mesh, shape=(batch_size,), replace=False)
    height_indices = selected_indices[:, 0]
    width_indices = selected_indices[:, 1]
    target_colors = image[height_indices, width_indices]

    def loss_fn(params):
        x = jnp.stack([height_indices/image_height, width_indices/image_width], axis=-1)
        predicted_colors = state.apply_fn({'params': params}, x)
        color_mse = jnp.mean((predicted_colors - target_colors)**2)
        return color_mse
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def create_train_state(model:nn.Module, learning_rate:float, KEY) -> TrainState:
    params = model.init(KEY, jnp.ones((1, 2)))['params']
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(learning_rate))

def render_image(state:TrainState, image_height:int, image_width:int, channels:int) -> jax.Array:
    height_indices, width_indices = jnp.meshgrid(
        jnp.arange(image_height)/image_height, 
        jnp.arange(image_width)/image_width
    )
    x = jnp.stack([height_indices.flatten(), width_indices.flatten()], axis=-1)
    predicted_colors = state.apply_fn({'params': state.params}, x)
    rendered_image = jnp.reshape(predicted_colors, (image_height, image_width, channels))
    rendered_image = jnp.transpose(rendered_image, (1, 0, 2))
    return rendered_image

def create_model_from_config(config:dict) -> NGPImage:
    model = NGPImage(
        number_of_grid_levels=config['num_hash_table_levels'],
        max_hash_table_entries=config['max_hash_table_entries'],
        hash_table_feature_dim=config['hash_table_feature_dim'],
        coarsest_resolution=config['coarsest_resolution'],
        finest_resolution=config['finest_resolution'],
        mlp_width=config['mlp_width'],
        mlp_depth=config['mlp_depth'],
        output_channels=config['channels']
    )
    return model

def move_pytree_to_cpu(pytree, cpu_id=0):
    device = jax.devices('cpu')[0]
    def move_to_cpu(tensor):
        return jnp.array(jax.device_put(tensor, device=device))
    return jax.tree_map(move_to_cpu, pytree)

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, default='configs/ngp_image.json')
    #args = parser.parse_args()

    with open('configs/ngp_image_robot_benchmark.json') as f:
        config = json.load(f)

    image_path = 'data/robot.jpg'
    image = Image.open(image_path)
    image = jnp.array(image) / 255.0
    print('Image shape:', image.shape)
    num_pixels = image.shape[0]*image.shape[1]
    print(f'Num pixels: {num_pixels:,}')

    model = create_model_from_config(config)
    state = create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f'Num params: {num_params:,}')
    print('Param to pixel ratio:', num_params/num_pixels)
    benchmark_train_loop(
        image=image, batch_size=config['batch_size'], 
        steps=config['train_steps'], state=state
    )
    
if __name__ == '__main__':
    main()
