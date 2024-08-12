import os
import sys
sys.path.append(os.getcwd())
import pytest
import jax
import jax.numpy as jnp
import fp_tokenization as fpt

def test_tokenization():
    batch = jax.random.normal(key=jax.random.PRNGKey(0), shape=(1, 256))
    tokens = fpt.tokenize(batch)
    print(tokens)
    print("Test didn't actually fail, but I'm failing it anyways to print the output")
    assert(False)
