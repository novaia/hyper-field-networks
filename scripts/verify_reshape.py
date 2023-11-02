import jax.numpy as jnp
import math

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
    bottom_padded = jnp.concatenate([bottom_padding, right_padded], axis=0)
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

def reshape_ae_output(x, square_pad_size, tile_pad_size, table_width):
    print(x.shape)
    num_tiles_per_dim = int(math.ceil(jnp.sqrt(x.shape[0])))
    x = jnp.split(x, num_tiles_per_dim, axis=0)
    vertically_joined = []
    for i in range(len(x)):
        vertically_joined.append(jnp.concatenate(x[i], axis=1))
    x = jnp.concatenate(vertically_joined, axis=0)
    x = jnp.squeeze(x, axis=-1)
    x = x[:-tile_pad_size, :-tile_pad_size]
    x = jnp.ravel(x)[:-square_pad_size]
    table_height = x.shape[0] // table_width
    hash_table = jnp.reshape(x, (table_height, table_width))
    return hash_table, table_height

def main():
    table_height = 524288
    table_width = 64
    tile_size = 128
    hash_table = jnp.arange(table_height * table_width)
    hash_table = jnp.reshape(hash_table, (table_height, table_width))
    original_hash_table = hash_table
    hash_table, num_tiles, square_pad_size, tile_pad_size = process_sample(
        hash_table, tile_size, table_height, return_padding=True
    )
    print('Autoencoded hash table', hash_table.shape)
    print(hash_table[0])
    hash_table, _ = reshape_ae_output(hash_table, square_pad_size, tile_pad_size, table_width)
    print(original_hash_table[:2])
    print(hash_table[:2])

if __name__ == '__main__':
    main()