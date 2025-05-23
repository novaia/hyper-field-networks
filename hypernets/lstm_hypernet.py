import os, json
import datasets
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
from fp_tokenization import get_vocab_size

def load_dataset(dataset_path, test_size, split_seed):
    field_config = None
    with open(os.path.join(dataset_path, 'field_config.json'), 'r') as f:
        field_config = json.load(f)
    assert field_config is not None
    param_map = None
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        param_map = json.load(f)
    assert param_map is not None
    
    parquet_dir = os.path.join(dataset_path, 'data')
    parquet_paths = [
        os.path.join(parquet_dir, p) 
        for p in os.listdir(parquet_dir) if p.endswith('.parquet')
    ]
    num_parquet_files = len(parquet_paths)
    assert num_parquet_files > 0
    print(f'Found {num_parquet_files} parquet file(s) in dataset directory')

    dataset = datasets.load_dataset('parquet', data_files=parquet_paths)
    train, test = dataset['train'].train_test_split(test_size=test_size, seed=split_seed).values()
    device = str(jax.devices('gpu')[0])
    train = train.with_format('jax', device=device)
    test = test.with_format('jax', device=device)
    context_length = train[0]['tokens'].shape[0]
    return train, test, field_config, param_map, context_length

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
        x = nn.remat(nn.Embed)(
            num_embeddings=self.vocab_size+1, 
            features=self.features, 
            dtype=self.dtype
        )(tokens)
        input_shape = x[:, 0].shape
        carry = lstm.initialize_carry(random.key(0), input_shape)
        carry, x = lstm(carry, x)
        x = nn.remat(nn.Dense)(self.vocab_size, dtype=self.dtype)(x)
        return x

@jax.jit
def train_step(state, tokens, start_tokens):
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1]], axis=-1)
    target_tokens = tokens
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, input_tokens)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, tokens, start_tokens):
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1]], axis=-1)
    target_tokens = tokens
    logits = state.apply_fn({'params': state.params}, tokens=input_tokens)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
    return loss

def main():
    output_path = 'data/ar_hypernet_output/6'
    dataset_path = 'data/mnist-ngp-image-612-16bit'
    split_size = 0.2
    split_seed = 0
    train_set, test_set, field_config, param_map, sequence_length = \
        load_dataset(dataset_path, split_size, split_seed)
    print('image width', field_config['image_width'])
    print('image height', field_config['image_height'])
    print('sequence length', sequence_length)
    
    vocab_size = get_vocab_size()
    print('vocab size', vocab_size)

    num_epochs = 10
    features = 512
    vocab_size = 2**16
    batch_size = 16
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

    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('Train set size:', num_train_samples)
    print('Train steps:', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('Test set size:', num_test_samples)
    print('Test steps:', test_steps)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        accumulated_losses = []
        for step in range(train_steps):
            tokens = next(train_iterator)['tokens']
            loss, state = train_step(state, tokens=tokens, start_tokens=batched_start_tokens)
            accumulated_losses.append(loss)
            print(f'step {step}, loss {loss}')
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            tokens = next(test_iterator)['tokens']
            loss = test_step(state, tokens=tokens, start_tokens=batched_start_tokens)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print('average test loss', average_test_loss)

if __name__ == '__main__':
    main()
