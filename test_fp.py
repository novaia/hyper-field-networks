import fp_conversion
from jax import numpy as jnp

tokens = fp_conversion.tokenize(jnp.arange(20, dtype=jnp.float16))
print(tokens)
samples = fp_conversion.detokenize(tokens)
print(samples)
