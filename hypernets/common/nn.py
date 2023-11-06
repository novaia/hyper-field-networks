from typing import Any
import flax.linen as nn
import jax.numpy as jnp
import jax

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
    embedding_min_frequency:float = 1.0
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

class LinearAttention(nn.Module):
    attention_dim:int
    output_dim:int
    normal_dtype:Any = jnp.float32
    quantized_dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        def qkv_projection(x):
            return nn.Dense(features=self.attention_dim, dtype=self.quantized_dtype)(x)
        query = qkv_projection(x)
        key = qkv_projection(x)
        value = qkv_projection(x)
        feature_map = lambda x: nn.elu(x) + 1.0
        Q = feature_map(query)
        K = feature_map(key)
        KV = jnp.einsum(
            'nsh,nsh->nh', 
            K.astype(self.normal_dtype), 
            value.astype(self.normal_dtype),
            precision='highest'
        )
        Z = 1/(jnp.einsum(
            'nlh,nh->nlh', 
            Q, jnp.sum(K, axis=1, dtype=self.quantized_dtype),
            precision='highest'
        ) + 1e-6)
        V = jnp.einsum("nlh,nh,nlh->nlh", Q, KV, Z)
        x = nn.Dense(features=self.output_dim, dtype=self.quantized_dtype)(V)
        x = nn.gelu(x)
        return x
    
class FeedForward(nn.Module):
    hidden_dim:int
    output_dim:int
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        return x

class LinearTransformer(nn.Module):
    num_blocks:int
    attention_dim:int
    residual_dim:int
    feed_forward_dim:int
    remat:bool = True
    normal_dtype:Any = jnp.float32
    quantized_dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.remat:
            CustomAttention = nn.remat(LinearAttention)
            CustomFeedForward = nn.remat(FeedForward)
        else:
            CustomAttention = LinearAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = CustomAttention(
                attention_dim=self.attention_dim, 
                output_dim=self.residual_dim,
                normal_dtype=self.normal_dtype, 
                quantized_dtype=self.quantized_dtype
            )(x)
            x = nn.LayerNorm(dtype=self.normal_dtype)((x + residual).astype(self.normal_dtype))
            residual = x
            x = CustomFeedForward(
                self.feed_forward_dim, self.residual_dim, self.quantized_dtype
            )(x)
            x = nn.LayerNorm(dtype=self.normal_dtype)((x + residual).astype(self.normal_dtype))
        return x

class VanillaTransformer(nn.Module):
    num_heads:int
    num_blocks:int
    attention_dim:int
    residual_dim:int
    feed_forward_dim:int
    remat:bool = False

    @nn.compact
    def __call__(self, x):
        if self.remat:
            CustomAttention = nn.remat(nn.SelfAttention)
            CustomFeedForward = nn.remat(FeedForward)
        else:
            CustomAttention = LinearAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = CustomAttention(
                num_heads=self.num_heads,
                qkv_features=self.attention_dim, 
                output_dim=self.residual_dim
            )(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = CustomFeedForward(self.feed_forward_dim, self.residual_dim)(x)
            x = nn.LayerNorm()(x + residual)
        return x