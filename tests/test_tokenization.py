import pytest
import jax
import jax.numpy as jnp
from float_tokenization import tokenize, detokenize, bit_count_to_vocab_size

@pytest.mark.parametrize('sequence_length', [128, 8192])
@pytest.mark.parametrize('bit_count', [11])
def test_token_pipeline(sequence_length, bit_count):
    vocab_size = bit_count_to_vocab_size(bit_count)
    # The tokenization is lossy so the tolerance is quite large.
    atol = 0.5
    input_samples = jax.random.normal(jax.random.PRNGKey(0), (sequence_length,))
    tokens = tokenize(input_samples)
    assert jnp.all(tokens < vocab_size), 'One or more tokens were greater than the vocab size.'
    assert jnp.all(tokens >= 0), 'One or more tokens were less than 0.'
    output_samples = detokenize(tokens)
    assert not jnp.any(jnp.isnan(output_samples)), 'One or more detokenized samples were NaN.'
    assert jnp.allclose(input_samples, output_samples, atol=atol), (
        'One or more detokenized samples were outside the tolerance range.'
    )
