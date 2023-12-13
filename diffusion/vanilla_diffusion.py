from typing import Any, Callable, List
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

class ResidualBlock(nn.Module):
    num_features: int
    num_groups: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb):
        input_width = x.shape[-1]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv(self.num_features, kernel_size=(1, 1))(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        time_emb = nn.Dense(self.num_features)(x)
        time_emb = self.activation_fn(time_emb)
        time_emb = jnp.expand_dims(time_emb, axis=(1, 2))
        time_emb = jnp.broadcast_to(time_emb, x.shape)
        x = x + time_emb
        x = nn.Conv(self.num_features, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features, 
                num_groups=self.num_groups,
                activation_fn=self.activation_fn
            )(x, time_emb)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips

class UpBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(
                num_features=self.num_features,
                num_groups=self.num_groups,
                activation_fn=self.activation_fn
            )(x, time_emb)
        return x, skips

class VanillaDiffusion(nn.Module):
    num_features: List[int]
    num_groups: List[int]
    activation_fn: Callable
    block_depth: int
    embedding_dim: int
    embedding_max_frequency: float
    output_channels: int

    @nn.compact
    def __call__(self, x):
        x, diffusion_time = x

        time_emb = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_max_frequency=self.embedding_max_frequency
        )(diffusion_time)

        skips = []
        features_and_groups = list(zip(self.num_features[:-1], self.num_groups[:-1]))
        for f, g in features_and_groups:
            x, skips = DownBlock(
                num_features=f,
                num_groups=g,
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)

        for _ in range(self.block_depth):
            x = ResidualBlock(
                num_features=self.num_features[-1],
                num_groups=self.num_groups[-1],
                activation_fn=self.activation_fn
            )(x)

        for f, g in reversed(features_and_groups):
            x, skips = UpBlock(
                num_features=f,
                num_groups=g,
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)

        x = nn.Conv(
            self.output_channels, 
            kernel_size=(1, 1), 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        return x