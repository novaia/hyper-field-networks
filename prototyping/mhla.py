import os
import sys
sys.path.append(os.getcwd())
from hypernets.common.nn import MultiHeadLinearAttention
import jax.numpy as jnp
import jax

x = jnp.ones((32, 128, 4))
key = jax.random.PRNGKey(0)
model = MultiHeadLinearAttention(attention_dim=64, output_dim=16, num_heads=4)
print(model.tabulate(key, x))