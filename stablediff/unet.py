import flax.linen as nn
import jax.numpy as jnp

from .attention_flax import FlaxTransformer2DModel
from .resnet_flax import Downsample2d, ResnetBlock2d, Upsample2d

# Refactored - done but untested
class CrossAttnDownBlock2d(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_downsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic:bool = True):
        output_states = ()
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            hidden_states = ResnetBlock2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )(hidden_states, temb, deterministic=deterministic)

            hidden_states = FlaxTransformer2DModel(
                in_channels=self.out_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.out_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
            )(hidden_states, encoder_hidden_states, deterministic=deterministic)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = Downsample2d(self.out_channels, dtype=self.dtype)(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states
    
# Refactored - done but untested
class DownBlock2d(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, temb, deterministic:bool = True):
        output_states = ()
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            hidden_states = ResnetBlock2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )(hidden_states, temb, deterministic=deterministic)
            output_states += (hidden_states,)
        
        if self.add_downsample:
            hidden_states = Downsample2d(self.out_channels, dtype=self.dtype)(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states

# Refactored - done but untested
class CrossAttnUpBlock2d(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_upsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False

    @nn.compact
    def __call__(
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, 
        deterministic:bool = True
    ):
        for i in range(self.num_layers):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
            
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
            hidden_states = ResnetBlock2d(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )(hidden_states, temb, deterministic=deterministic)

            hidden_states = FlaxTransformer2DModel(
                in_channels=self.out_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.out_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
            )(hidden_states, encoder_hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = Upsample2d(self.out_channels, dtype=self.dtype)(hidden_states)
        return hidden_states
    
# Refactored - done but untested
class UpBlock2d(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, res_hidden_states_tuple, temb, deterministic=True):
        for i in range(self.num_layers):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            if i == self.num_layers - 1:
                res_skip_channels = self.in_channels
            else:
                res_skip_channels = self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
            hidden_states = ResnetBlock2d(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )(hidden_states, temb, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = Upsample2d(self.out_channels, dtype=self.dtype)(hidden_states)
        return hidden_states

# Refactored - done but untested
class UNetMidBlock2dCrossAttn(nn.Module):
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True):
        # there is always at least one resnet
        hidden_states = ResnetBlock2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            dropout_prob=self.dropout,
            dtype=self.dtype,
        )(hidden_states, temb)

        for _ in range(self.num_layers-1):
            hidden_states = FlaxTransformer2DModel(
                in_channels=self.in_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.in_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                dtype=self.dtype,
            )(hidden_states, encoder_hidden_states, deterministic=deterministic)

            hidden_states = ResnetBlock2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )(hidden_states, temb, deterministic=deterministic)

        return hidden_states
