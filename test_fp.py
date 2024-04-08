from float_tokenization import tokenize, detokenize 
from jax import numpy as jnp

tokens = tokenize(jnp.arange(20, dtype=jnp.float16))
print(tokens)
samples = detokenize(tokens)
print(samples)
