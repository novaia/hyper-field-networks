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

# Replaced by AttentionBlock
class FlaxAttentionBlock(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

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

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        # compute attentions
        attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
        attention_scores = attention_scores * self.scale
        attention_probs = nn.softmax(attention_scores, axis=2)

        # attend to values
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
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

# Replaced by BasicTransformerBlock
class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxAttentionBlock(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        # cross attention
        self.attn2 = FlaxAttentionBlock(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.ff = FlaxGluFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
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

# Replaced by Transformer2dModel
class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.proj_out(hidden_states)

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

# Replaced by GluFeedForward
class FlaxGluFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a gated linear unit activation from:
    https://arxiv.org/abs/2002.05202

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
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

# Replaced by GegluFeedForward
class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return hidden_linear * nn.gelu(hidden_gelu)