import flax.linen as nn
import jax.numpy as jnp

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
    embedding_min_frequency:float = 1.0

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis = -1
        )
        return embeddings

class LinearAttention(nn.Module):
    attention_dim:int
    output_dim:int

    @nn.compact
    def __call__(self, x):
        query = nn.Dense(features=self.attention_dim)(x)
        key = nn.Dense(features=self.attention_dim)(x)
        value = nn.Dense(features=self.attention_dim)(x)
        feature_map = lambda x: nn.elu(x) + 1.0
        Q = feature_map(query)
        K = feature_map(key)
        KV = jnp.einsum('nsh,nsh->nh', K, value)
        Z = 1/(jnp.einsum("nlh,nh->nlh", Q, jnp.sum(K, axis=1))+1e-6)
        V = jnp.einsum("nlh,nh,nlh->nlh", Q, KV, Z)
        x = nn.Dense(features=self.output_dim)(V)
        x = nn.gelu(x)
        return x
    
class FeedForward(nn.Module):
    hidden_dim:int
    output_dim:int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.gelu(x)
        return x

class LinearTransformer(nn.Module):
    num_blocks:int
    attention_dim:int
    residual_dim:int
    feed_forward_dim:int
    remat:bool = True

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
            x = CustomAttention(self.attention_dim, self.residual_dim)(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = CustomFeedForward(self.feed_forward_dim, self.residual_dim)(x)
            x = nn.LayerNorm()(x + residual)
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
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = CustomFeedForward(self.feed_forward_dim, self.residual_dim)(x)
            x = nn.LayerNorm()(x + residual)
        return x