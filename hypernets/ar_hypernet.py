import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
import datasets
import os, json
from typing import Any
from functools import partial
from hypernets.common.nn import SinusoidalEmbedding, kl_divergence
from float_tokenization import detokenize
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt

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
    return train.with_format('jax'), test.with_format('jax'), field_config, param_map

class ArHypernet(nn.Module):
    vocab_size: int
    context_length: int
    embedding_dim: int
    hidden_dim: int
    num_attention_heads: int
    num_blocks: int
    dtype: Any

    @nn.compact
    def __call__(self, tokens, attention_mask):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(tokens)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.embedding_dim)(positions)
        x = x + pos_emb
        x = nn.Dense(features=self.hidden_dim)(x)
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.hidden_dim,
                out_features=self.hidden_dim,
                dtype=self.dtype
            )(inputs_q=x, mask=attention_mask)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
            x = x + residual
        
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits

@jax.jit
def train_step(state, tokens, attention_mask):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, tokens, attention_mask)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, tokens, attention_mask):
    logits = state.apply_fn({'params': state.params}, tokens, attention_mask)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
    return loss

def sample_context(state, prompt_token, context_length, temperature=1.0):
    tokens_to_sample = context_length - 1
    filler_tokens = jnp.zeros((tokens_to_sample), dtype=jnp.uint32)
    tokens = jnp.array([prompt_token], dtype=jnp.uint32)
    tokens = jnp.concatenate((tokens, filler_tokens), axis=-1)
    tokens = jnp.expand_dims(tokens, axis=0)
    attention_mask = nn.make_causal_mask(tokens, dtype=jnp.bfloat16)

    @jax.jit
    def get_logits(state, tokens, attention_mask):
        return state.apply_fn({'params': state.params}, tokens, attention_mask)

    for i in range(tokens_to_sample):
        logits = get_logits(state, tokens, attention_mask)
        logits = logits[:, i, :] / temperature
        probs = nn.softmax(logits)
        next_token = jax.random.categorical(jax.random.PRNGKey(i), probs, shape=(1,))
        next_token = jnp.array(next_token, dtype=jnp.uint32)[0]
        tokens = tokens.at[0, i+1].set(next_token)
    
    return tokens

def main():
    output_path = 'data/ar_hypernet_output/0'
    dataset_path = 'data/mnist-ngp-image-612-11bit'
    split_size = 0.01
    split_seed = 0
    train_set, test_set, field_config, param_map = load_dataset(dataset_path, split_size, split_seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    token_bits = 11
    vocab_size = 2 * 2**token_bits - 1
    print('vocab size', vocab_size)

    num_epochs = 200
    batch_size = 16
    context_length = 612
    embedding_dim = 256
    hidden_dim = 512
    num_attention_heads = 16
    num_blocks = 12
    learning_rate = 1e-4
    weight_decay = 1e-3

    model = ArHypernet(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        num_blocks=num_blocks,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length), dtype=jnp.uint32)
    attention_mask = nn.make_causal_mask(x, dtype=jnp.bfloat16)
    print('mask', attention_mask.shape)
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x, attention_mask)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('test set size:', num_train_samples)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('train set size:', num_test_samples)
    
    prompt_token = test_set[0]['tokens'][0]
    print('sample prompt token:', prompt_token)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            tokens = next(train_iterator)['tokens']
            loss, state = train_step(state, tokens, attention_mask)
            losses_this_epoch.append(loss)
            #print(f'step {step}, loss {loss}')
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            tokens = next(test_iterator)['tokens']
            loss = test_step(state, tokens, attention_mask)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print(f'epoch {epoch}, test_loss {average_test_loss}')
        
        tokens = sample_context(state, prompt_token, context_length)[0]
        flat_params = detokenize(tokens)
        # Detokenized NaNs should not exist so there is probably a bug in the detokenization code.
        flat_params = jnp.nan_to_num(flat_params)

        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), field_render)

if __name__ == '__main__':
    main()
