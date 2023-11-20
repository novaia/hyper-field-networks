import jax
import jax.numpy as jnp

def sample_normals(means, standard_deviations, key):
    def sample(mean, std, key):
        return std * jax.random.normal(key) + mean
    split_keys = jax.random.split(key, means.shape[-1])
    return jax.vmap(sample, in_axes=0)(means, standard_deviations, split_keys)

key = jax.random.PRNGKey(3)
means = jnp.array([0.2, 0.3, 1.0, 0.9])
means = jnp.repeat(jnp.expand_dims(means, axis=0), 20, axis=0)
deviations = jnp.array([0.4, 0.1, 2.0, 0.01])
deviations = jnp.repeat(jnp.expand_dims(deviations, axis=0), 20, axis=0)
split_keys = jax.random.split(key, means.shape[0])
samples = jax.vmap(sample_normals, in_axes=0)(means, deviations, split_keys)
print(jnp.mean(samples, axis=0))
print(jnp.std(samples, axis=0))