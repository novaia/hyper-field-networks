# Image Field Diffusion Transformer.
# The purpose of this script is to test a full-attention diffusion transformer
# on the task of image field synthesis.
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from hypernets.common.nn import VanillaTransformer
from typing import Any, Callable

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

class DiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    embedding_dim:int
    num_blocks:int
    feed_forward_dim:int
    token_dim:int
    context_length:int
    activation_fn:Callable
    normal_dtype:Any
    quantized_dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x, t):
        positions = jnp.arange(self.context_length+1)
        position_embedding = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        position_embedding = jnp.expand_dims(position_embedding, axis=0)
        
        time_embedding = SinusoidalEmbedding(
            self.embedding_dim,
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(t)
        #print('time_embedding', time_embedding.shape)
        time_embedding = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.Dense(
            self.embedding_dim,
            dtype=self.quantized_dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.LayerNorm()(time_embedding)
        
        skip = x
        skip_weight = nn.Dense(
            self.embedding_dim,
            dtype=self.quantized_dtype
        )(time_embedding)
        skip_weight = self.activation_fn(skip_weight)
        skip_weight = nn.Dense(1, dtype=self.quantized_dtype)(skip_weight)
        skip_weight = nn.sigmoid(skip_weight)

        #print('x', x.shape)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.quantized_dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.LayerNorm()(x)
        #print('x', x.shape)

        # Add the diffusion time token to the end of the sequence.
        x = jnp.concatenate([x, time_embedding], axis=-2)
        x = x + position_embedding
        #print('x', x.shape)

        x = VanillaTransformer(
            num_blocks=self.num_blocks,
            attention_dim=self.attention_dim,
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim,
            feed_forward_dim=self.feed_forward_dim,
            activation_fn=self.activation_fn,
            normal_dtype=self.normal_dtype,
            quantized_dtype=self.quantized_dtype,
            remat=self.remat
        )(x)

        # Remove the diffusion time token from the end of the sequence.
        x = x[:, :-1, :]
        x = nn.Dense(
            features=self.token_dim, dtype=self.quantized_dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        x = x * (1 - skip_weight) + skip * (skip_weight)
        #print('output', x.shape)
        return x
