import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax

def normalize(x:jax.Array, eps:float = 1e-12, axis:int = 1):
    return x / jnp.clip(jnp.linalg.norm(x, axis=axis), a_min=eps)

# Style mapper M which maps latent code z and text descriptor t_global to style vector w.
class StyleMapper(nn.Module):
    # Mapping Network M layer depth.
    depth: int
    # w dimension.
    dim: int

    @nn.compact
    def __call__(self, x):
        x = normalize(x, axis=1)
        for _ in range(self.depth):
            x = nn.Dense(self.dim)(x)
            # TODO: verify leaky relu slope, default is 0.01 but lucidrains
            # uses something different
            x = nn.leaky_relu(x)
        return x

class LearnedTextEncoder(nn.Module):
    # Text Transformer T layer depth.
    depth: int
    dim: int
    num_heads: int = 1

    @nn.compact
    def __call__(self, z, t_global):
        x = jnp.concatenate([z, t_global], axis=-1)
        for _ in range(self.depth):
            # Does this need to be L2 attention like the rest of the network?
            # Should this include residual connections?
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.dim
            )(x)
            x = nn.Dense(self.dim)(x)
            x = nn.leaky_relu(x)
            x = nn.Dense(self.dim)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.dim)(x)
        t_local = x[:, :-1, :]
        t_global = x[:, -1, :]
        return t_local, t_global

class AdaptiveConv(nn.Module):
    num_filter_blocks: int
    num_filters: int
    kernel_size: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x, w):
        channels_in = x.shape[-1]
        filter_bank = self.param(
            'filter_bank', 
            nn.initializers.normal(),
            (
                self.num_filter_blocks, 
                self.num_filters, 
                channels_in,
                self.kernel_size, 
                self.kernel_size
            )
        )
        filter_block_weights = nn.softmax(nn.Dense(self.num_filter_blocks)(x))
        # Notation: b = filter blocks, f = filters, i = input channells, 
        # h = kernel height, w = kernel width, n = batch
        filters = jnp.einsum('bfchw,nb->nfchw', filter_bank, filter_block_weights)
        modulation_scales = nn.Dense(self.num_filters)(w)
        filters = jnp.einsum('nfchw,nf...->nfchw', filter, modulation_scales+1)
        filterwise_norm = jnp.sqrt(
            jnp.clamp(jnp.einsum('nfchw,nfchw->nhw', filters, filters), a_min=self.eps)
        )
        filters = jnp.einsum('nfchw,n...hw->nfchw', filters, 1.0 / filterwise_norm)

class Generator(nn.Module):
    pretrained_text_encoder: nn.Module
    text_encoder_depth: int
    text_encoder_dim: int
    style_mapper_depth: int
    style_dim: int # w dimension
    num_filters: int
    features_per_filter: int
    kernel_size: int

    @nn.compact
    def __call__(self, z, c):
        t = self.pretrained_text_encoder(c)
        t_local, t_global = LearnedTextEncoder(
            depth=self.text_encoder_depth,
            dim=self.text_encoder_dim
        )(t)
        w = StyleMapper(depth=self.style_mapper_depth, dim=self.style_dim)(z, t_global)
