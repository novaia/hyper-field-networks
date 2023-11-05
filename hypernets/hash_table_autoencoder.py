import jax
import jax.numpy as jnp
import random
import os
import json
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import List
import optax
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image

def get_file_names(path:str):
    file_names = os.listdir(path)
    valid_file_names = []
    for name in file_names:
        if name.endswith('.npy'):
            valid_file_names.append(os.path.join(path, name))
    return valid_file_names

def square_reshape(array):
    if len(array.shape) != 1:
        raise ValueError("The array must be 1D")
    n = array.shape[0]
    k = int(jnp.ceil(jnp.sqrt(n)))
    pad_size = k**2 - n
    new_array = jnp.concatenate([array, jnp.zeros(pad_size)])
    new_array = jnp.reshape(new_array, (k, k))
    return new_array, pad_size

def split_into_tiles(array, tile_size):
    if len(array.shape) != 2:
        raise ValueError("The array must be 2D")
    pad_size = (tile_size - (array.shape[0] % tile_size)) % tile_size
    right_padding = jnp.zeros((array.shape[0], pad_size))
    right_padded = jnp.concatenate([array, right_padding], axis=1)
    bottom_padding = jnp.zeros((pad_size, right_padded.shape[1]))
    bottom_padded = jnp.concatenate([right_padded, bottom_padding], axis=0)
    num_splits = bottom_padded.shape[0] // tile_size
    vertical_split = jnp.split(bottom_padded, num_splits, axis=0)
    vertical_split = jnp.stack(vertical_split)
    horizontal_split = jnp.split(vertical_split, num_splits, axis=2)
    horizontal_split = jnp.concatenate(horizontal_split, axis=0)
    num_tiles = horizontal_split.shape[0]
    return horizontal_split, num_tiles, pad_size

def process_sample(sample, tile_size, table_height, return_padding=False):
    sample = sample[:table_height]
    sample = jnp.ravel(sample)
    sample, square_pad_size = square_reshape(sample)
    sample, num_tiles, tile_pad_size = split_into_tiles(sample, tile_size)
    sample = jnp.expand_dims(sample, axis=-1)
    if return_padding:
        return sample, num_tiles, square_pad_size, tile_pad_size
    return sample, num_tiles

def load_batch(file_names:list, batch_size:int, table_height:int, tile_size:int, key):
    file_names = random.sample(file_names, 1)
    num_loaded_samples = 0
    batch = []
    for name in file_names:
        sample = jnp.load(name)
        sample, num_tiles = process_sample(sample, tile_size, table_height)
        batch.append(sample)
        num_loaded_samples += num_tiles
        if num_loaded_samples >= batch_size:
            break
    batch = jnp.concatenate(batch, axis=0)
    batch_indices = jax.random.choice(
        key, jnp.arange(batch.shape[0]), shape=(batch_size,), replace=False
    )
    return batch[batch_indices]

class HashTableAutoencoder(nn.Module):
    widths: List[int]
    kernel_size = (3, 3)
    pool_size = (2, 2)
    upsample_size = 2
    
    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.Conv(width, self.kernel_size)(x)
            x = nn.GroupNorm()(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=self.pool_size, strides=self.pool_size)
        for width in reversed(self.widths):
            upsample_shape = (
                x.shape[0], 
                x.shape[1] * self.upsample_size, 
                x.shape[2] * self.upsample_size, 
                x.shape[3]
            )
            x = jax.image.resize(x, upsample_shape, method='nearest')
            x = nn.Conv(width, self.kernel_size)(x)
            x = nn.GroupNorm()(x)
            x = nn.relu(x)
        x = nn.Conv(1, self.kernel_size)(x)
        x = nn.relu(x)
        return x

def train_loop(state, steps, get_batch_fn, batch_size):
    for step in range(steps):
        batch = get_batch_fn(batch_size=batch_size, key=jax.random.PRNGKey(step))
        loss, state = train_step(state, batch)
        print(f'Step {step}, loss {loss}')
    return state

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        return jnp.mean((state.apply_fn({'params': params}, batch) - batch)**2)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def create_train_state(rng, model, input_shape, learning_rate):
    x = jnp.ones(input_shape)
    params = model.init(rng, x)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def main():
    dataset_path = 'data/synthetic_nerfs/packed_aliens'
    learning_rate = 1e-4
    batch_size = 32
    widths = [128, 128]
    train_steps = 300
    tile_size = 128

    file_names = get_file_names(dataset_path)
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        table_metadata = json.load(f)[-1]
        table_height = table_metadata['table_height']

    get_batch_fn = partial(
        load_batch, file_names=file_names, table_height=table_height, tile_size=tile_size
    )
    batch = get_batch_fn(batch_size=batch_size, key=jax.random.PRNGKey(0))
    model = HashTableAutoencoder(widths)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, batch.shape, learning_rate)
    state = train_loop(state, train_steps, get_batch_fn, batch_size)

    test_sample_full = jnp.load(file_names[0])
    test_sample, num_tiles, square_pad_size, tile_pad_size = process_sample(
        test_sample_full, tile_size, table_height, return_padding=True
    )
    batched_test_sample = jnp.array_split(test_sample, batch_size, axis=0)
    batched_output = []
    for i in range(len(batched_test_sample)):
        batched_output.append(state.apply_fn({'params': state.params}, batched_test_sample[i]))
    batched_output = jnp.concatenate(batched_output, axis=0)

    autoencoder_output = {
        'output': batched_output, 
        'square_pad_size': square_pad_size, 
        'tile_pad_size': tile_pad_size,
        'input_path': file_names[0] 
    }
    jnp.save('data/autoencoder_output.npy', autoencoder_output, allow_pickle=True)

if __name__ == '__main__':
    main()