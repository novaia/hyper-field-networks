from float_tokenization import tokenize, detokenize 
from jax import numpy as jnp
import jax

samples_in = jax.random.normal(jax.random.PRNGKey(0), (15000,), dtype=jnp.float16)
print(samples_in)
tokens = tokenize(samples_in)
print(tokens)
samples = detokenize(tokens)
print(samples)
