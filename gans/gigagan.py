import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax
from typing import Tuple, List, Union
from functools import partial

def normalize(x:jax.Array, eps:float = 1e-12, axis:int = 1):
    return x / jnp.clip(jnp.linalg.norm(x, axis=axis, keepdims=True), a_min=eps)

# All patch functions assume that the channel axis comes before the spatial axes.
# They also assume that the image tensor is square, meaning height = width.
#@partial(jax.jit, static_argnames=['image_size', 'patch_size'])
def precompute_patch_data(image_size, patch_size):
    num_patches_across = image_size // patch_size
    num_patches_total = num_patches_across**2
    indices = jnp.arange(num_patches_across) * patch_size
    horizontal_indices, vertical_indices = jnp.meshgrid(indices, indices)
    return num_patches_across, num_patches_total, horizontal_indices, vertical_indices

@partial(jax.jit, static_argnames=['num_patches_total', 'patch_size', 'channels'])
def patchify(x, num_patches_total, horizontal_indices, vertical_indices, patch_size, channels):
    def get_patch(x, horizontal_id, vertical_id):
        start_indices = [horizontal_id, vertical_id, 0]
        slice_sizes = [patch_size, patch_size, channels]
        patch = lax.dynamic_slice(x, start_indices, slice_sizes)
        return jnp.ravel(patch)

    patches = jax.vmap(
        jax.vmap(get_patch, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
    )(x, horizontal_indices, vertical_indices)
    patches = jnp.reshape(patches, (num_patches_total, patch_size*patch_size*channels))
    return patches

@partial(jax.jit, static_argnames=[
    'patch_size', 'num_patches_across', 'num_patches_total', 'channels'
])
def depatchify(x, patch_size, num_patches_across, num_patches_total, channels):
    x = jnp.reshape(x, (num_patches_total, patch_size, patch_size, channels))
    x = jnp.array(jnp.split(x, num_patches_across, axis=0))
    x = jax.vmap(lambda a: jnp.concatenate(a, axis=0), in_axes=0)(x)
    x = jnp.concatenate(x, axis=1)
    return x

# Swaps channel and spatial axes of an image tensor so that channel axis comes first.
# NHWC -> NCWH -> NCHW.
def swap_channel_first(x):
    return jnp.swapaxes(jnp.swapaxes(x, 1, 3), 2, 3)

# Swaps channel and spatial axes of an image tensor so that spatial axes come first.
# NCHW -> NCWH -> NHWC.
def swap_spatial_first(x):
    return jnp.swapaxes(jnp.swapaxes(x, 3, 1), 2, 1)

# The GigaGAN paper uses L2 distance attention instead of dot product attention because it has 
# a better Lipschitz constant which is important for discriminator stability. 
@partial(jax.vmap, in_axes=0)
def l2_attention(key, query, value):
    l2_squared = jax.vmap(
        jax.vmap(
            lambda k, q: jnp.sum((q - k)**2, axis=-1), 
            in_axes=(0, None)
        ), 
        in_axes=(None, 0)
    )(key, query)
    # TODO: scale down l2 distance logits.
    weights = nn.softmax(-l2_squared)
    # TODO: verify that reduction is on the correct axis.
    return weights.T @ value

# Style mapper M which maps latent code z and text descriptor t_global to style vector w.
class StyleMapper(nn.Module):
    depth: int
    dim: int

    @nn.compact
    def __call__(self, z, t_global):
        x = jnp.concatenate([z, t_global], axis=-1)
        x = normalize(x, axis=1)
        for _ in range(self.depth):
            x = nn.Dense(self.dim)(x)
            # TODO: verify leaky relu slope, default is 0.01 but lucidrains
            # uses something different
            x = nn.leaky_relu(x)
        return x

# TODO: implement CLIP.
class PretrainedTextEncoder(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, c):
        return jnp.ones((c.shape[0], c.shape[1], self.dim))

class LearnedTextEncoder(nn.Module):
    # Text Transformer T layer depth.
    depth: int
    dim: int
    num_heads: int = 1

    @nn.compact
    def __call__(self, t):
        for _ in range(self.depth):
            # Does this need to be L2 attention like the rest of the network?
            # Should this include residual connections?
            # = jnp.expand_dims(t, axis=-1)
            t = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.dim
            )(inputs_q=t, inputs_kv=t)
            #t = jnp.squeeze(axis=-1)
            t = nn.Dense(self.dim)(t)
            t = nn.leaky_relu(t)
            t = nn.Dense(self.dim)(t)
            t = nn.leaky_relu(t)
        t = nn.Dense(self.dim)(t)
        t_local = t[:, :-1, :]
        t_global = t[:, -1, :]
        return t_local, t_global

class AdaptiveConv(nn.Module):
    num_filters: int
    num_features: int
    kernel_size: int
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x, w):
        channels_in = x.shape[1]
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
        filter_weights = nn.softmax(nn.Dense(self.num_filters)(w))
        # Notation: b = filters, f = features, c = input channels, 
        # h = kernel height, w = kernel width, n = batch
        ada_filter = jnp.einsum('bfchw,nb->nfchw', filter_bank, filter_weights)
        modulation_scales = nn.Dense(self.num_features)(w)
        ada_filter = jnp.einsum('nfchw,nf...->nfchw', ada_filter, modulation_scales+1)
        filterwise_norm = jnp.sqrt(
            jnp.clip(jnp.einsum('nfchw,nfchw->nhw', ada_filter, ada_filter), a_min=self.eps)
        )
        ada_filter = jnp.einsum('nfchw,n...hw->nfchw', ada_filter, 1.0 / filterwise_norm)
        
        batch_size = w.shape[0]
        input_height = x.shape[-2]
        input_width = x.shape[-1]

        # TODO: verify correctness of grouped convolutions.
        # nchw - > 1(nc)hw
        grouped_x_shape = (1, batch_size * channels_in, input_height, input_width)
        x = jnp.reshape(x, grouped_x_shape)
        # nfchw -> (nf)chw
        grouped_filter_shape = (
            batch_size * self.num_features, 
            channels_in, 
            self.kernel_size, 
            self.kernel_size)
        ada_filter = jnp.reshape(ada_filter, grouped_filter_shape)
        # lhs should be nchw, rhs should be (nf)chw for groups = n
        x = lax.conv_general_dilated(
            lhs=x, 
            rhs=ada_filter, 
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
        new_shape = (x.shape[0], x.shape[1], x.shape[2]*self.factor, x.shape[3]*self.factor)
        return jax.image.resize(x, new_shape, self.method)

class SelfAttention(nn.Module):
    attention_dim: int
    num_heads: int
    output_dim: int

    @nn.compact
    def __call__(self, x, w):
        assert self.attention_dim % self.num_heads == 0, (
            f'attention_dim {self.attention_dim} is not divisible by num_heads '
            f'{self.num_heads}'
        )

        head_dim = self.attention_dim // self.num_heads
        def qkv_projection(x, name):
            return nn.DenseGeneral(axis=-1, features=(self.num_heads, head_dim), name=name)(x)

        w = nn.Dense(x.shape[-1])(w)
        w = jnp.expand_dims(w, axis=1)
        x = jnp.concatenate([x, w], axis=1)

        # (batch, context, tokens) -> (batch, context, num_heads, head_dim).
        key = qkv_projection(x, 'key')
        query = qkv_projection(x, 'query')
        value = qkv_projection(x, 'value')
        # Map across multi-head dimension.
        x = jax.vmap(l2_attention, in_axes=-2)(key, query, value)
        # Remove style token.
        x = x[:, :, :-1, :]
        # (num_heads, batch, context, head_dim) -> (batch, context, output_dim).
        x = nn.DenseGeneral(features=self.output_dim, axis=(0, -1), name='out')(x)
        x = nn.leaky_relu(x)
        return x
    
class CrossAttention(nn.Module):
    attention_dim: int
    num_heads: int
    output_dim: int

    @nn.compact
    def __call__(self, x, t_local):
        assert self.attention_dim % self.num_heads == 0, (
            f'attention_dim {self.attention_dim} is not divisible by num_heads '
            f'{self.num_heads}'
        )
        
        head_dim = self.attention_dim // self.num_heads
        def qkv_projection(x, name):
            return nn.DenseGeneral(axis=-1, features=(self.num_heads, head_dim), name=name)(x)
        
        # (batch, context, tokens) -> (batch, context, num_heads, head_dim).
        key = qkv_projection(x, 'key')
        query = qkv_projection(t_local, 'query')
        value = qkv_projection(t_local, 'value')
        # Map across multi-head dimension.
        x = jax.vmap(l2_attention, in_axes=-2)(key, query, value)
        # (num_heads, batch, context, head_dim) -> (batch, context, output_dim).
        x = nn.DenseGeneral(features=self.output_dim, axis=(0, -1), name='out')(x)
        x = nn.leaky_relu(x)
        return x

class SynthesisBlockAttention(nn.Module):
    depth: int
    patch_size: int

    @nn.compact
    def __call__(self, x, w, t_local):
        image_size = x.shape[-1]
        channels = x.shape[1]
        # Patch functions expect spatial axes to come before the channel axis.
        x = swap_spatial_first(x)
        num_patches_across, num_patches_total, horizontal_indices, vertical_indices = (
            precompute_patch_data(image_size=image_size, patch_size=self.patch_size)
        )
        # TODO: why can't I use kwargs when vmapping? Is this a bug?
        x = jax.vmap(patchify, in_axes=(0, None, None, None, None, None))(
            x, 
            num_patches_total, 
            horizontal_indices,
            vertical_indices,
            self.patch_size,
            channels
        )
        attention_dim = x.shape[-1]
        num_heads = 1
        output_dim = attention_dim
        x = SelfAttention(attention_dim, num_heads, output_dim)(x, w)
        x = CrossAttention(attention_dim, num_heads, output_dim)(x, t_local)
        x = jax.vmap(depatchify, in_axes=(0, None, None, None, None))(
            x,
            self.patch_size,
            num_patches_across,
            num_patches_total,
            channels
        )
        x = swap_channel_first(x)
        return x

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
        x = self.param(
            'learned_constant', 
            nn.initializers.normal(),
            (1, self.image_channels, self.lowest_resolution, self.lowest_resolution)
        )
        # Repeat the learned constant along the batch dimension.
        x = jnp.repeat(x, w.shape[0], axis=0)

        attention_depth_id = 0
        image_pyramid = []
        pyramid_levels = len(self.synthesis_block_depths)
        current_resolution = self.lowest_resolution
        
        for l in range(pyramid_levels):
            attention_depth = None
            if current_resolution in self.attention_resolutions:
                attention_depth = self.attention_depths[attention_depth_id]
                attention_depth_id += 1
            
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
            if l < pyramid_levels - 1:
                x = UpSample()(x)
                current_resolution *= 2
        return image_pyramid
    
def main():
    key = jax.random.PRNGKey(0)

    batch_size = 32
    latent_dim = 128
    max_prompt_tokens = 77

    pretrained_text_encoder = PretrainedTextEncoder(768)
    text_encoder_depth = 4
    text_encoder_dim = 768
    style_mapper_depth = 4
    style_dim = 1024
    lowest_resolution = 4
    g_attention_resolutions = [8, 16, 32]
    g_attention_depths = [2, 2, 1]
    g_attention_dim = 0
    patch_size = 4
    synthesis_block_depths = [3, 3, 3, 2, 2]
    g_filters_per_level = [1, 1, 2, 4, 8]
    g_num_conv_features = 2
    kernel_size = 3
    image_channels = 3

    generator = Generator(
        pretrained_text_encoder=pretrained_text_encoder,
        text_encoder_depth=text_encoder_depth,
        text_encoder_dim=text_encoder_dim,
        style_mapper_depth=style_mapper_depth,
        style_dim=style_dim,
        lowest_resolution=lowest_resolution,
        attention_resolutions=g_attention_resolutions,
        attention_depths=g_attention_depths,
        patch_size=patch_size,
        synthesis_block_depths=synthesis_block_depths,
        filters_per_level=g_filters_per_level,
        num_conv_features=g_num_conv_features,
        kernel_size=kernel_size,
        image_channels=image_channels
    )

    key, z_key, c_key = jax.random.split(key, 3)
    z = jax.random.normal(z_key, (batch_size, latent_dim))
    c = jax.random.normal(c_key, (batch_size, max_prompt_tokens, text_encoder_dim))

    key, model_key = jax.random.split(key, 2)
    print(generator.tabulate(model_key, z, c))

if __name__ == '__main__':
    main()