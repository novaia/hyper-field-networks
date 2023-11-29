import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax
from typing import Tuple, List, Union

def normalize(x:jax.Array, eps:float = 1e-12, axis:int = 1):
    return x / jnp.clip(jnp.linalg.norm(x, axis=axis), a_min=eps)

def patchify(x, patch_size):
    return x

def depatchify(x):
    return x

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
    num_filters: int
    num_features: int
    kernel_size: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x, w):
        channels_in = x.shape[-1]
        filter_bank = self.param(
            'filter_bank', 
            nn.initializers.normal(),
            (
                self.num_filters, 
                self.num_features, 
                channels_in,
                self.kernel_size, 
                self.kernel_size
            )
        )
        filter_block_weights = nn.softmax(nn.Dense(self.num_filters)(x))
        # Notation: b = filter blocks, f = filters, i = input channells, 
        # h = kernel height, w = kernel width, n = batch
        filters = jnp.einsum('bfchw,nb->nfchw', filter_bank, filter_block_weights)
        modulation_scales = nn.Dense(self.num_features)(w)
        filters = jnp.einsum('nfchw,nf...->nfchw', filter, modulation_scales+1)
        filterwise_norm = jnp.sqrt(
            jnp.clamp(jnp.einsum('nfchw,nfchw->nhw', filters, filters), a_min=self.eps)
        )
        filters = jnp.einsum('nfchw,n...hw->nfchw', filters, 1.0 / filterwise_norm)
        
        batch_size = filters.shape[0]
        input_height = x.shape[-2]
        input_width = x.shape[-1]

        # TODO: verify correctness of grouped convolutions.
        # nchw - > 1(nc)hw
        grouped_x_shape = (1, batch_size * channels_in, input_height, input_width)
        x = jnp.reshape(x, grouped_x_shape)
        # nfchw -> (nf)chw
        grouped_filters_shape = (
            batch_size * self.num_features, 
            channels_in, 
            self.kernel_size, 
            self.kernel_size)
        filters = jnp.reshape(filters, grouped_filters_shape)
        # lhs should be nchw, rhs should be (nf)chw for groups = n
        x = lax.conv_general_dilated(
            lhs=x, 
            rhs=filters, 
            feature_group_count=batch_size, 
            padding='SAME', 
            window_strides=(1, 1)
        )
        x = jnp.reshape(x, (batch_size, self.num_features, input_height, input_width))
        return x

class UpSample(nn.Module):
    factor: int = 2
    method: str = 'bilinear'

    @nn.compact
    def __call__(self, x):
        new_shape = (x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor, x.shape[3])
        return jax.image.resize(x, new_shape, self.method)

class SelfAttention(nn.Module):
    @nn.compact
    def __call__(self, x, w):
        return x
    
class CrossAttention(nn.Module):
    @nn.compact
    def __call__(self, x, t_local):
        return x

class SynthesisBlockAttention(nn.Module):
    depth: int
    patch_size: int

    @nn.compact
    def __call__(self, x, w, t_local):
        x = patchify(x, self.patch_size)
        x = SelfAttention()(x, w)
        x = CrossAttention()(x, t_local)
        x = depatchify(x)

class SynthesisBlock(nn.Module):
    depth: int
    num_filters: int
    num_conv_features: int
    kernel_size: int
    image_channels: int
    patch_size: int
    attention_depth: Union[None, int]

    @nn.compact
    def __call__(self, x, w, t_local):
        for _ in range(self.depth):
            x = AdaptiveConv(self.num_filters, self.num_conv_features, self.kernel_size)(x, w)
            if self.attention_depth is not None:
                x = SynthesisBlockAttention(
                    depth=self.attention_depth, patch_size=self.patch_size
                )(x, w, t_local)
        kernel_size = (self.kernel_size, self.kernel_size)
        image = nn.Conv(self.image_channels, kernel_size=kernel_size)(x)
        image = nn.sigmoid(image)
        return x, image

class Generator(nn.Module):
    pretrained_text_encoder: nn.Module
    # In the paper: text transformer T layer depth.
    text_encoder_depth: int
    text_encoder_dim: int
    # In the paper: mapping network M layer depth.
    style_mapper_depth: int
    # In the paper: w dimension.
    style_dim: int
    lowest_resolution: int 
    
    # In the paper: G attention resolutions.
    # The resolutions at which attention should be applied.
    attention_resolutions: List[int]
    # In the paper: G attention depths.
    attention_depths: List[int]
    # Patch size for image tokenization. Token dimension will be patch_size^2.
    patch_size: int

    # In the paper: num synthesis block per resolution.
    # len(synthesis_block_depths) = pyramid_levels = L.
    synthesis_block_depths: List[int]
    # In the paper: G num filters N for adaptive kernel selection.
    filters_per_level: List[int]
    num_conv_features: int
    kernel_size: int
    image_channels: int

    @nn.compact
    def __call__(self, z, c):
        t = self.pretrained_text_encoder(c)
        t_local, t_global = LearnedTextEncoder(
            depth=self.text_encoder_depth,
            dim=self.text_encoder_dim
        )(t)
        w = StyleMapper(depth=self.style_mapper_depth, dim=self.style_dim)(z, t_global)
        x = self.params(
            'learned_constant', 
            nn.initializers.normal(), 
            (self.lowest_resolution, self.lowest_resolution, self.image_channels)
        )

        attention_depth_id = 0
        image_pyramid = []
        pyramid_levels = len(self.synthesis_block_depths)
        current_resolution = self.lowest_resolution
        
        for l in range(pyramid_levels):
            attention_depth = None
            if current_resolution in self.attention_resolutions:
                attention_depth = self.attention_depths[attention_depth_id]
                attention_resolution_id += 1
            
            x, image = SynthesisBlock(
                depth=self.synthesis_block_depths[l],
                num_filters=self.filters_per_level[l],
                num_conv_features=self.num_conv_features,
                kernel_size=self.kernel_size,
                image_channels=self.image_channels,
                patch_size=self.patch_size,
                attention_depth=attention_depth
            )(x, w, t_local)
            
            image_pyramid.append(image)
            if l < self.pyramid_levels - 1:
                x = UpSample()
                current_resolution *= 2

        return image_pyramid
