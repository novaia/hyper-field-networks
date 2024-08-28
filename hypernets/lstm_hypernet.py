import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import matplotlib.pyplot as plt
import optax
from functools import partial
from typing import Any

class LSTM(nn.Module):
    features: int
    vocab_size: int
    dtype: Any

    @nn.compact
    def __call__(self, tokens):
        ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=1, out_axes=1
        )        
        lstm = ScanLSTM(self.features, dtype=self.dtype)
        x = nn.Embed(num_embeddings=self.vocab_size+1, features=self.features, dtype=self.dtype)(tokens)
        input_shape = x[:, 0].shape
        print(input_shape)
        carry = lstm.initialize_carry(random.key(0), input_shape)
        carry, x = lstm(carry, x)
        x = nn.remat(nn.Dense)(self.vocab_size, dtype=self.dtype)(x)
        return x

@jax.jit
def train_step(state, tokens, start_tokens):
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1]], axis=-1)
    target_tokens = tokens
    def loss_fn(params):
        logits = state.apply_fn(params, input_tokens)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def main():
    features = 128
    vocab_size = 2**16
    batch_size = 4
    sequence_length = 18000
    learning_rate = 3e-4
    start_token = vocab_size
    model_dtype = jnp.bfloat16
    batched_start_tokens = jnp.full((batch_size, 1), fill_value=start_token, dtype=jnp.uint32)
    
    model = LSTM(features=features, vocab_size=vocab_size, dtype=model_dtype)
    x = jnp.ones((batch_size, sequence_length), dtype=jnp.uint32)
    params_key = jax.random.PRNGKey(204)
    params = jax.jit(model.init)(params_key, tokens=x)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adam(learning_rate=learning_rate)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

if __name__ == '__main__':
    main()
