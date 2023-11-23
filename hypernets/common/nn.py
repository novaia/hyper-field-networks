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
    num_heads:int
    normal_dtype:Any = jnp.float32
    quantized_dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        scaled_attention_dim = self.attention_dim // self.num_heads
        def qkv_projection(x):
            return nn.Dense(features=scaled_attention_dim, dtype=self.quantized_dtype)(x)
        attention_head_outputs = []
        for i in range(self.num_heads):
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
            attention_head_outputs.append(V)
        x = jnp.concatenate(attention_head_outputs, axis=-1)
        x = nn.Dense(features=self.output_dim, dtype=self.quantized_dtype)(x)
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
                num_heads=self.num_attention_heads,
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
            CustomAttention = nn.SelfAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = CustomAttention(
                num_heads=self.num_heads,
                qkv_features=self.attention_dim, 
                out_features=self.residual_dim
            )(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = CustomFeedForward(self.feed_forward_dim, self.residual_dim)(x)
            x = nn.LayerNorm()(x + residual)
        return x

# Linear Attention Diffusion Transformer = LADiT.
class LinearAttentionDiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int
    normal_dtype:Any
    quantized_dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x):
        CustomDense = nn.remat(nn.Dense) if self.remat else nn.Dense
        x, noise_variances = x
        e = SinusoidalEmbedding(
            self.token_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(noise_variances)
        x = jnp.concatenate([x, e], axis=-2)
        x = CustomDense(features=self.embedding_dim, dtype=self.quantized_dtype)(x)
        positions = jnp.arange(self.context_length+1)
        e = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        x = x + e

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
        x = x[:, :-1, :]
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
            x = LinearAttention(
                attention_dim=dim, 
                output_dim=dim,
                num_heads=self.num_attention_heads
            )(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x + residual)

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
            x = LinearAttention(
                attention_dim=dim, 
                output_dim=dim,
                num_heads=self.num_attention_heads
            )(x)
            x = nn.LayerNorm()(x + residual)
            residual = x
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x + residual)

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