from typing import Any
import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.linear import PrecisionLike

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

def linear_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    dtype: Any,
    precision: PrecisionLike = None,
):
    feature_map = lambda x: nn.elu(x) + 1.0
    Q = feature_map(query)
    K = feature_map(key)
    KV = jnp.einsum(
        '...sh,...sh->...h', 
        K.astype(dtype), 
        value.astype(dtype),
        precision=precision
    )
    Z = 1/(jnp.einsum(
        '...lh,...h->...lh', 
        Q, jnp.sum(K, axis=1, dtype=dtype),
        precision=precision
    ) + 1e-6)
    return jnp.einsum("...lh,...h,...lh->...lh", Q, KV, Z)

class MultiHeadLinearAttention(nn.Module):
    attention_dim:int
    output_dim:int
    num_heads:int
    normal_dtype:Any = jnp.float32
    quantized_dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        assert self.attention_dim % self.num_heads == 0, (
            'Attention dim {self.attention_dim} must be divisible by number of heads '
            '{self.num_heads}'
        )
        head_dim = self.attention_dim // self.num_heads
        def qkv_projection(x, name):
            return nn.DenseGeneral(
                axis=-1, features=(self.num_heads, head_dim), 
                dtype=self.quantized_dtype, name=name
            )(x)
        query = qkv_projection(x, 'query')
        key = qkv_projection(x, 'key')
        value = qkv_projection(x, 'value')
        x = jax.vmap(linear_attention, in_axes=(-2, -2, -2, None, None))(
            query, key, value, self.quantized_dtype, 'highest'
        )
        # TODO: double check that these are the correct axes. 
        # There might be cross batch contamination.
        x = nn.DenseGeneral(features=self.output_dim, axis=(0, -1), name='out')(x)
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
    num_attention_heads:int
    residual_dim:int
    feed_forward_dim:int
    remat:bool = True
    normal_dtype:Any = jnp.float32
    quantized_dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.remat:
            CustomAttention = nn.remat(MultiHeadLinearAttention)
            CustomFeedForward = nn.remat(FeedForward)
        else:
            CustomAttention = MultiHeadLinearAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm(dtype=self.normal_dtype)(x.astype(self.normal_dtype))
            x = CustomAttention(
                attention_dim=self.attention_dim, 
                output_dim=self.residual_dim,
                num_heads=self.num_attention_heads,
                normal_dtype=self.normal_dtype, 
                quantized_dtype=self.quantized_dtype
            )(x)
            x = x + residual
            residual = x
            x = nn.LayerNorm(dtype=self.normal_dtype)(x.astype(self.normal_dtype))
            x = CustomFeedForward(
                self.feed_forward_dim, self.residual_dim, self.quantized_dtype
            )(x)
            x = x + residual
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
            CustomAttention = nn.SelfAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = CustomAttention(
                num_heads=self.num_heads,
                qkv_features=self.attention_dim, 
                out_features=self.residual_dim
            )(x)
            x = nn.gelu(x)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = CustomFeedForward(self.feed_forward_dim, self.residual_dim)(x)
            x = x + residual
        return x

# Linear Attention Diffusion Transformer = LADiT.
class LinearAttentionDiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    token_dim:int
    embedding_max_frequency:float
    context_length:int
    normal_dtype:Any
    quantized_dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x, t):
        half_embedding_dim = self.embedding_dim // 2
        x = nn.Dense(half_embedding_dim)(x)
        time_embedding = SinusoidalEmbedding(
            half_embedding_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(t)
        # Add the diffusion time token to the end of the sequence.
        x = jnp.concatenate([x, time_embedding], axis=-2)

        positions = jnp.arange(self.context_length+1)
        position_emb = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=half_embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        position_emb = jnp.broadcast_to(position_emb, x.shape)
        x = jnp.concatenate([x, position_emb], axis=-1)

        x = LinearTransformer(
            num_blocks=self.num_bocks, 
            attention_dim=self.attention_dim, 
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype,
            remat=self.remat
        )(x)

        # Remove the diffusion time token from the end of the sequence.
        x = x[:, :-1, :]
        
        CustomDense = nn.remat(nn.Dense) if self.remat else nn.Dense
        x = CustomDense(
            features=self.token_dim, dtype=self.quantized_dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        return x

Ladit = LinearAttentionDiffusionTransformer

class AutoPositionalEmbedding(nn.Module):
    num_positions: int
    feature_dim: int

    @nn.compact
    def __call__(self, x):
        positions = jnp.arange(self.num_positions)
        e = nn.Embed(self.num_positions, self.feature_dim)(positions)
        return x + e

class TransformerVaeEncoder(nn.Module):
    context_length: int
    hidden_dims: int
    num_attention_heads: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = AutoPositionalEmbedding(self.context_length, x.shape[-1])(x)
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            residual = x
            x = nn.LayerNorm()(x)
            x = MultiHeadLinearAttention(
                attention_dim=dim, 
                output_dim=dim,
                num_heads=self.num_attention_heads
            )(x)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = x + residual

        flattened_shape = (x.shape[0], self.context_length * x.shape[-1])
        x = jnp.reshape(x, flattened_shape)
        means = nn.Dense(self.latent_dim)(x)
        logvars = nn.Dense(self.latent_dim)(x)
        return means, logvars
    
class TransformerVaeDecoder(nn.Module):
    context_length: int
    hidden_dims: int
    num_attention_heads: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.context_length * self.hidden_dims[0])(x)
        tokenized_shape = (x.shape[0], self.context_length, self.hidden_dims[0])
        x = jnp.reshape(x, tokenized_shape)
        x = AutoPositionalEmbedding(self.context_length, x.shape[-1])(x)

        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            residual = x
            x = nn.LayerNorm()(x)
            x = MultiHeadLinearAttention(
                attention_dim=dim, 
                output_dim=dim,
                num_heads=self.num_attention_heads
            )(x)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = x + residual

        logits = nn.Dense(self.output_dim)(x)
        return logits
    
class TransformerVae(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x):
        x, key = x
        means, logvars = self.encoder(x)
        x = means + jnp.exp(0.5 * logvars) * jax.random.normal(key, means.shape)
        logits = self.decoder(x)
        return logits, means, logvars
    
@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))
