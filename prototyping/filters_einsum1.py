import jax.numpy as jnp
import jax
import numpy as np

num_filters = 20
features_per_filter = 4
kernel_size = 3

filter_bank = jnp.arange(num_filters*features_per_filter*kernel_size*kernel_size)
filter_bank = jnp.reshape(
    filter_bank, 
    newshape=(num_filters, features_per_filter, kernel_size, kernel_size)
)
print(filter_bank.shape)

key = jax.random.PRNGKey(0)
batch_size = 16
weights = jax.nn.softmax(jax.random.normal(key, (batch_size, num_filters)))

simple_sum = np.zeros((batch_size, features_per_filter, kernel_size, kernel_size))
for b in range(batch_size):
    for i in range(num_filters):
        simple_sum[b] += weights[b, i] * filter_bank[i]
simple_sum = jnp.array(simple_sum)

print(simple_sum.shape)

einstein_sum = jnp.einsum('bfhw,nb->nfhw', filter_bank, weights)
print(einstein_sum.shape)

print(einstein_sum[0, 0])
print(simple_sum[0, 0])


print(jnp.allclose(simple_sum, einstein_sum))