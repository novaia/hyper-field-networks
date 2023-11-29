import jax
import jax.numpy as jnp
from jax import lax

num_filter_blocks = 64
num_filters = 8
channels_in = 3
kernel_size = 6
batch_size = 4
eps = 1e-8
image_height = 32
image_width = 32

key = jax.random.PRNGKey(0)
filter_bank_shape = (
    num_filter_blocks,
    num_filters,
    channels_in,
    kernel_size,
    kernel_size
)
filter_bank = jax.random.normal(key, filter_bank_shape)

filter_block_weights = jax.nn.softmax(jnp.ones((batch_size, num_filter_blocks)))
# Notation: b = filter blocks, f = filters, c = input channels, 
# h = kernel height, w = kernel width, n = batch
filters = jnp.einsum('bfchw,nb->nfchw', filter_bank, filter_block_weights)
print('weighted filters', filters.shape)
modulation_scales = jnp.ones((batch_size, num_filters))
filters = jnp.einsum('nfchw,nf...->nfchw', filters, modulation_scales+1)
print('modulated filters', filters.shape)
filterwise_norm = jnp.sqrt(
    jnp.clip(jnp.einsum('nfchw,nfchw->nhw', filters, filters), a_min=eps)
)
print('filterwise norm', filterwise_norm.shape)
filters = jnp.einsum('nfchw,n...hw->nfchw', filters, 1.0 / filterwise_norm)
print('demodulated filters', filters.shape)
x = jnp.ones((batch_size, channels_in, image_height, image_width))
print('x', x.shape)
# TODO: figure out how to implement grouped convolutions
# lhs should be nchw, rhs should be fchw
# y = lax.conv(lhs=x, rhs=filters, padding='same', window_strides=(1, 1))
# print('y', y.shape)