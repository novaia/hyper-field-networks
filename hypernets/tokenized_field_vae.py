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

def load_dataset(dataset_path):
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

    dataset = datasets.load_dataset(
        'parquet', 
        data_files={'train': parquet_paths},
        split='train',
        num_proc=1
    )
    dataset = dataset.with_format('jax')
    return dataset, field_config, param_map

class TokenizedFieldVae(nn.Module):
    vocab_size: int
    context_length: int
    embedding_dim: int
    latent_dim: int
    num_attention_heads: int
    num_blocks: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, tokens, key):
        def transformer_block(x):
            for _ in range(self.num_blocks):
                residual = x
                x = nn.LayerNorm(dtype=self.dtype)(x)
                x = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_attention_heads,
                    qkv_features=self.embedding_dim,
                    out_features=self.embedding_dim,
                    dtype=self.dtype
                )(inputs_q=x)
                x = x + residual
                residual = x
                x = nn.LayerNorm(dtype=self.dtype)(x)
                x = nn.Dense(features=self.embedding_dim, dtype=self.dtype)(x)
                x = nn.gelu(x)
                x = nn.Dense(features=self.embedding_dim, dtype=self.dtype)(x)
                x = nn.gelu(x)
                x = x + residual
                return x

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(tokens)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.embedding_dim)(positions)
        x = x + pos_emb
        x = transformer_block(x)
        means = nn.DenseGeneral(features=self.latent_dim, axis=(-1, -2), dtype=self.dtype)(x)
        logvars = nn.DenseGeneral(features=self.latent_dim, axis=(-1, -2), dtype=self.dtype)(x)
        x = means + jnp.exp(0.5 * logvars) * jax.random.normal(key, means.shape)
        x = nn.DenseGeneral(features=(self.context_length, self.embedding_dim), axis=-1, dtype=self.dtype)(x)
        x = transformer_block(x)
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits, means, logvars

def make_kl_schedule(initial_value, final_value, transition_steps, cycle_steps):
    slope = final_value / transition_steps
    linear_fn = partial(
        lambda m, b, max_y, x: jnp.clip(m * x + b, 0, max_y), 
        slope, initial_value, final_value
    )
    cyclical_fn = partial(lambda period, x: linear_fn(x % period), cycle_steps)
    return cyclical_fn

@jax.jit
def train_step(state, tokens, kl_weight):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits, means, logvars = state.apply_fn({'params': params}, tokens, key)
        ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
        kld_loss = jnp.mean(kl_divergence(means, logvars))
        loss = ce_loss + (kld_loss * kl_weight)
        return loss, (ce_loss, kld_loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, kld_loss)), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return (loss, ce_loss, kld_loss), state

def main():
    dataset_path = 'data/mnist-ngp-image-612-11bit'
    dataset, field_config, param_map = load_dataset(dataset_path)
    
    token_bits = 11
    vocab_size = 2 * 2**token_bits - 1
    print('vocab size', vocab_size)

    num_epochs = 30
    batch_size = 16
    context_length = 612
    embedding_dim = 128
    latent_dim = 512
    num_attention_heads = 4
    num_blocks = 8
    learning_rate = 3e-4
    
    model = TokenizedFieldVae(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        num_attention_heads=num_attention_heads,
        num_blocks=num_blocks,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length), dtype=jnp.uint32)
    vae_key = jax.random.PRNGKey(68)
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x, vae_key)['params']
    opt = optax.adam(learning_rate=learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    num_samples = len(dataset)
    steps_per_epoch = num_samples // batch_size
    
    cycle_steps = 4 * steps_per_epoch
    kl_weight_schedule = make_kl_schedule(
        initial_value=0.0,
        final_value=0.0,
        transition_steps=cycle_steps,
        cycle_steps=cycle_steps
    )

    for epoch in range(num_epochs):
        dataset.shuffle(seed=epoch)
        dataset_iterator = dataset.iter(batch_size)
        for step in range(steps_per_epoch):
            tokens = next(dataset_iterator)['tokens']
            kl_weight = kl_weight_schedule(state.step)
            (loss, ce_loss, kld_loss), state = train_step(state, tokens, kl_weight)
            print(f'step {step}, loss {loss}, ce_loss {ce_loss}, kld_loss {kld_loss}')

if __name__ == '__main__':
    main()
