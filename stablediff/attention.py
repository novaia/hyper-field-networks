import flax.linen as nn
import jax.numpy as jnp

# Refactored - done but untested
class AttentionBlock(nn.Module):
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, context=None, deterministic:bool = True):
        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = jnp.transpose(tensor, (0, 2, 1, 3))
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor
        
        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = jnp.transpose(tensor, (0, 2, 1, 3))
            tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor
        
        context = hidden_states if context is None else context
        inner_dim = self.dim_head * self.heads
        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        query_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, name="to_q"
        )(hidden_states)
        key_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, name="to_k"
        )(context)
        value_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, name="to_v"
        )(context)

        query_states = reshape_heads_to_batch_dim(query_proj)
        key_states = reshape_heads_to_batch_dim(key_proj)
        value_states = reshape_heads_to_batch_dim(value_proj)

        # compute attentions
        attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
        attention_scores = attention_scores * self.scale
        attention_probs = nn.softmax(attention_scores, axis=2)

        # attend to values
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
        hidden_states = reshape_batch_dim_to_heads(hidden_states)
        hidden_states = nn.Dense(
            self.query_dim, dtype=self.dtype, name="to_out_0"
        )(hidden_states)
        return hidden_states
    
# Refactored - done but untested
class BasicTransformerBlock(nn.Module):
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, context, deterministic=True):
        # self attention (or cross_attention if only_cross_attention is True)
        residual = hidden_states
        hidden_states = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(hidden_states)
        attention1 = AttentionBlock(
            self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype
        )
        if self.only_cross_attention:
            hidden_states = attention1(hidden_states, context, deterministic=deterministic)
        else:
            hidden_states = attention1(hidden_states, context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(hidden_states)
        hidden_states =  AttentionBlock(
            self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype
        )(hidden_states, context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        hidden_states = GluFeedForward(
            dim=self.dim, dropout=self.dropout, dtype=self.dtype
        )(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states
    
# Refactored - done but untested
# A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
# https://arxiv.org/pdf/1506.02025.pdf
class Transformer2dModel(nn.Module):
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, context, deterministic:bool = True):
        inner_dim = self.n_heads * self.d_head
        batch, height, width, channels = hidden_states.shape
        
        # project in
        residual = hidden_states
        hidden_states = nn.GroupNorm(num_groups=32, epsilon=1e-5)(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = nn.Dense(inner_dim, dtype=self.dtype)(hidden_states)
        else:
            hidden_states = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        # transformer
        for _ in range(self.depth):
            hidden_states = BasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
            )(hidden_states)

        # project out
        if self.use_linear_projection:
            hidden_states = nn.Dense(inner_dim, dtype=self.dtype)(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states

# Refactored - done but untested
class GluFeedForward(nn.Module):
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic=True):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        hidden_states = GegluFeedForward(self.dim, self.dropout, self.dtype, name='net_0')(hidden_states)
        hidden_states = nn.Dense(self.dim, dtype=self.dtype, name='net_2')(hidden_states)
        return hidden_states

# Refactored - done but untested
# Linear layer folled by the variant of the gated linear unit activation function from:
# https://arxiv.org/abs/2002.05202
class GegluFeedForward(nn.Module):
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def __call__(self, hidden_states, deterministic=True):
        inner_dim = self.dim * 4
        hidden_states = nn.Dense(inner_dim * 2, dtype=self.dtype)(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return hidden_linear * nn.gelu(hidden_gelu)
