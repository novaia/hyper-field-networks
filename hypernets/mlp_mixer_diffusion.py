# The purpose of this script is to see whether or not conditioning an image DiT
# with labels (i.e. MNIST digit labels) significantly improves performance.
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from hypernets.common.nn import VanillaTransformer
from matplotlib import pyplot as plt
import wandb
import pandas as pd

from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types

from functools import partial
from typing import Any, Callable
import json
import os

class Mlp(nn.Module):
    dim: int
    depth: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(features=self.dim, dtype=self.dtype)(x)
            x = self.activation_fn(x)
        return x

class AdaLayerNorm(nn.Module):
    dim: int
    depth: int
    activation_fn: Callable
    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, ada_input):
        x = nn.LayerNorm(
            use_bias=False, use_scale=False, 
            epsilon=self.epsilon, dtype=self.dtype
        )(x)
        ada_params = Mlp(
            dim=self.dim, depth=self.depth, 
            activation_fn=self.activation_fn, dtype=self.dtype
        )(ada_input)
        ada_params = nn.Dense(features=2, dtype=self.dtype)(x)
        scale = ada_params[..., 0:1]
        bias = ada_params[..., 1:2]
        x = x * (1 + scale) + bias
        return x

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_min_frequency:float = 1.0
    embedding_max_frequency:float = 1000.0
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
                dtype=self.dtype
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis = -1,
            dtype=self.dtype
        )
        return embeddings

class DiffusionMlpMixer(nn.Module):
    dim: int
    context_length: int
    output_dim: int
    num_labels: int
    num_blocks: int
    mlp_depth: int
    ada_ln_depth: int
    activation_fn: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t, label):
        t_emb = SinusoidalEmbedding(embedding_dim=self.dim, dtype=self.dtype)(t)
        label_emb = nn.Embed(
            num_embeddings=self.num_labels,
            features=self.dim
        )(jnp.ravel(label))

        x = Mlp(
            dim=self.dim, depth=3, 
            activation_fn=self.activation_fn, dtype=self.dtype
        )(x)
        label_emb = jnp.expand_dims(label_emb, axis=1)
        x = jnp.concatenate([x, label_emb], axis=-2)
        num_tokens = self.context_length + 1

        def norm(x):
            return AdaLayerNorm(
                dim=self.dim, depth=self.ada_ln_depth, 
                activation_fn=self.activation_fn, dtype=self.dtype
            )(x, t_emb)

        def transpose(x):
            return jnp.swapaxes(x, -1, -2)

        def mlp(x, transposed):
            dim = self.dim if not transposed else num_tokens
            return Mlp(
                dim=dim, depth=self.mlp_depth, 
                activation_fn=self.activation_fn, dtype=self.dtype
            )(x)

        for _ in range(self.num_blocks):
            residual = x
            x = norm(x)
            x = transpose(x)
            x = mlp(x, transposed=True)
            x = transpose(x)
            x = x + residual
            residual = x
            x = norm(x)
            x = mlp(x, transposed=False)
            x = x + residual

        x = nn.Dense(
            features=self.output_dim, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)

def main():
    batch_size = 4
    context_length = 1024
    model = DiffusionMlpMixer(
        dim=128,
        context_length=context_length,
        output_dim=1,
        ada_ln_depth=1,
        num_labels=10,
        num_blocks=16,
        mlp_depth=3,
        activation_fn=nn.gelu,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length, 1))
    t = jnp.ones((batch_size, 1))
    labels = jnp.ones((batch_size, 1), dtype=jnp.uint8)
    params = model.init(jax.random.PRNGKey(121), x, t, labels)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adaptive_grad_clip(clipping=3.0),
        optax.adam(learning_rate=3e-4)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'Param count: {param_count:,}')

if __name__ == '__main__':
    main()
