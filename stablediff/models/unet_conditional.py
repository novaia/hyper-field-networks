from typing import Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Dict

from .embedding import TimestepEmbedding, SinusoidalTimestepEmbedding
from .output import BaseOutput
from .unet import (
    CrossAttnDownBlock2d,
    CrossAttnUpBlock2d,
    DownBlock2d,
    UNetMidBlock2dCrossAttn,
    UpBlock2d,
)

# Refactored - done but untested
@flax.struct.dataclass
class UNet2dConditionalOutput(BaseOutput):
    sample: jnp.ndarray

# Refactored - done but untested
# UNet2dConditionalModel is a conditional 2D UNet model that takes in a noisy sample, 
# conditional state, and a timestep and returns sample shaped output.
class UNet2dConditionalModel(nn.Module):
    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    # TODO: switch these to enums because strings are too error prone.
    down_block_types: Tuple[str] = (
        "CrossAttnDownBlock2d",
        "CrossAttnDownBlock2d",
        "CrossAttnDownBlock2d",
        "DownBlock2d",
    )
    up_block_types: Tuple[str] = (
        "UpBlock2d", 
        "CrossAttnUpBlock2d", 
        "CrossAttnUpBlock2d", 
        "CrossAttnUpBlock2d"
    )
    only_cross_attention: Union[bool, Tuple[bool]] = False
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int]] = 8
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0

    def init_weights(self, rng: jax.random.PRNGKey) -> Dict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, sample, timesteps, encoder_hidden_states)["params"]

    @nn.compact
    def __call__(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[UNet2dConditionalOutput, Tuple]:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4
        
        only_cross_attention = self.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(self.down_block_types)

        attention_head_dim = self.attention_head_dim
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(self.down_block_types)

        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        t_emb = SinusoidalTimestepEmbedding(
            block_out_channels[0], 
            flip_sin_to_cos=self.flip_sin_to_cos, 
            freq_shift=self.freq_shift
        )(timesteps)
        t_emb = TimestepEmbedding(time_embed_dim, dtype=self.dtype)(t_emb)

        # 2. pre-process
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(sample)

        # 3. down
        down_block_res_samples = (sample,)
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2d":
                sample, res_samples = CrossAttnDownBlock2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    attn_num_head_channels=attention_head_dim[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    dtype=self.dtype,
                )(sample, t_emb, encoder_hidden_states, deterministic=not train)
            else:
                sample, res_samples = DownBlock2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )(sample, t_emb, deterministic=not train)
            down_block_res_samples += res_samples

        # 4. mid
        sample = UNetMidBlock2dCrossAttn(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            attn_num_head_channels=attention_head_dim[-1],
            use_linear_projection=self.use_linear_projection,
            dtype=self.dtype,
        )(sample, t_emb, encoder_hidden_states, deterministic=not train)

        # 5. up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[: -(self.layers_per_block + 1)]

            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            if up_block_type == "CrossAttnUpBlock2d":
                sample = CrossAttnUpBlock2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    attn_num_head_channels=reversed_attention_head_dim[i],
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    dtype=self.dtype,
                )(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
            else:
                sample = UpBlock2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )(
                    sample, 
                    temb=t_emb, 
                    res_hidden_states_tuple=res_samples, 
                    deterministic=not train
                )

        # 6. post-process
        sample = nn.GroupNorm(num_groups=32, epsilon=1e-5)(sample)
        sample = nn.silu(sample)
        sample = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))

        if not return_dict:
            return (sample,)
        return UNet2dConditionalOutput(sample=sample)