import jax.numpy as jnp
import math

autoencoder_output = jnp.load('data/autoencoder_output.npy', allow_pickle=True).tolist()
x = autoencoder_output['output']
square_pad_size = autoencoder_output['square_pad_size']
tile_pad_size = autoencoder_output['tile_pad_size']
print('square_pad_size', square_pad_size)
print('tile_pad_size', tile_pad_size)
print(x.shape)
num_tiles_per_dim = int(math.ceil(jnp.sqrt(x.shape[0])))
x = jnp.split(x, num_tiles_per_dim, axis=0)
vertically_joined = []
for i in range(len(x)):
    vertically_joined.append(jnp.concatenate(x[i], axis=1))
x = jnp.concatenate(vertically_joined, axis=0)
x = jnp.squeeze(x, axis=-1)
x = x[:-tile_pad_size, :-tile_pad_size]
print(x.shape)
x = jnp.ravel(x)[:-square_pad_size]
table_width = 64
table_height = x.shape[0] // table_width
x = jnp.reshape(x, (table_height, table_width))
print(x.shape)