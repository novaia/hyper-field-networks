import os
import sys
sys.path.append(os.getcwd())
import pytest
import jax
import jax.numpy as jnp
import fp_tokenization as fpt

def test_tokenization():
    initial_batch = jax.random.normal(key=jax.random.PRNGKey(0), shape=(1, 256))
    tokens = fpt.tokenize(initial_batch)
    detokenized_batch = fpt.detokenize(tokens)
    print(initial_batch)
    print(detokenized_batch)
    assert jnp.allclose(initial_batch, detokenized_batch, atol=1e-3)

def test_bitfield():
    initial_batch = jax.random.normal(key=jax.random.PRNGKey(0), shape=(256,))
    bitfields = fpt.fp32_to_bitfield16(initial_batch)
    recovered_batch = fpt.bitfield16_to_fp32(bitfields)
    print(initial_batch)
    print(bitfields)
    print(recovered_batch)
    print(jnp.max(initial_batch - recovered_batch))
    assert jnp.allclose(initial_batch, recovered_batch, atol=1e-3)

def test_u8_tokenization():
    initial_batch = jax.random.normal(key=jax.random.PRNGKey(0), shape=(1, 256))
    tokens = fpt.u8_tokenize(initial_batch)
    print(tokens.dtype)
    assert tokens.dtype == jnp.uint8
    detokenized_batch = fpt.u8_detokenize(tokens)
    print(tokens)
    print(initial_batch)
    print(detokenized_batch)
    # Tolerance has to be very high since fp8 loses a lot of precision.
    assert jnp.allclose(initial_batch, detokenized_batch, atol=4e-1)

def test_vocab_size():
    expected_vocab_size = 2**16
    computed_vocab_size = fpt.get_vocab_size()
    assert computed_vocab_size == expected_vocab_size
