import fp_conversion
from jax import numpy as jnp

print(fp_conversion.tokenize(jnp.arange(20, dtype=jnp.float16)))
