import flax.linen as nn
import jax
import jax.numpy as jnp

# Refactored - done but untested
class Upsample2d(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            name='conv'
        )(hidden_states)
        return hidden_states
    
# Refactored - done but untested
class Downsample2d(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        # hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # padding="VALID",
            dtype=self.dtype,
            name='conv'
        )(hidden_states)
        return hidden_states
    
# Refactored - done but untested
class ResnetBlock2d(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, temb, deterministic=True):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        residual = hidden_states
        hidden_states = nn.GroupNorm(num_groups=32, epsilon=1e-5, name='norm1')(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            name='conv1'
        )(hidden_states)

        temb = nn.Dense(out_channels, dtype=self.dtype, name='time_emb_proj')(nn.swish(temb))
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        hidden_states = hidden_states + temb

        hidden_states = nn.GroupNorm(num_groups=32, epsilon=1e-5, name='norm2')(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = nn.Dropout(self.dropout_prob)(hidden_states, deterministic)
        hidden_states = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            name='conv2'
        )(hidden_states)

        shortcut = self.use_nin_shortcut
        if shortcut is None:
            shortcut = self.in_channels != out_channels
        if shortcut:
            residual = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
                name='conv3'
            )(residual)
        return hidden_states + residual