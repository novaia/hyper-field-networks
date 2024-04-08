import jax
from jax import numpy as jnp
from flax import linen as nn
import datasets
import os, json
from typing import Any

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
        x = transformer_block(x)
        means = nn.DenseGeneral(features=self.latent_dim, axis=(-1, -2), dtype=self.dtype)(x)
        logvars = nn.DenseGeneral(features=self.latent_dim, axis=(-1, -2), dtype=self.dtype)(x)
        x = means + jnp.exp(0.5 * logvars) * jax.random.normal(key, means.shape)
        x = nn.DenseGeneral(features=(self.context_length, self.embedding_dim), axis=-1, dtype=self.dtype)(x)
        x = transformer_block(x)
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits

def main():
    dataset_path = 'data/mnist-ngp-image-612-11bit'
    dataset, field_config, param_map = load_dataset(dataset_path)
    
    token_bits = 11
    vocab_size = 2 * 2**token_bits - 1
    print('vocab size', vocab_size)

    batch_size = 16
    context_length = 612
    embedding_dim = 64
    latent_dim = 512
    num_attention_heads = 4
    num_blocks = 8
    
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

if __name__ == '__main__':
    main()
